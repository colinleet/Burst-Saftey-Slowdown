using AudioAnalysis.DSPLib;
using AudioAnalysis.Levels123;
using Cysharp.Threading.Tasks;
using LeetUtils.NativeJobs;
using LeetUtils.Extensions;
using QuickStopwatch;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace AudioAnalysis.Nodes {

    /// <summary>
    /// This is the process which perofrms FFTs on raw channel data.
    /// </summary>
    public class FFTProcessRunner : MonoBase_IAnalaysisJob<float, float> {

        #region Managed Data

        /// <summary>
        /// Flag marks if FFT spectrum data assemblers are being used for this FFT.
        /// These classes only need to be used when not using the job based systems.
        /// </summary>
        private bool usingSpectrumDataAssembler = false;

        /// <summary>
        /// Contains the spectrum data magnitudes
        /// </summary>
        private FFTSpectrumDataAssembler spectrumData;

        /// <summary>
        /// This container holds FFTData converted to DBs
        /// </summary>
        private FFTSpectrumDataAssembler spectrumDataDBV;

        private bool managingSpectData_NA = false;

        private NativeArray<float> spectDataNA;

        private bool managingSpectDataDVB_NA = false;

        private NativeArray<float> spectDataDBV_NA;

        /// <summary>
        /// Are we currently running scheduled (unmanaged) jobs which have to be completed before the application closes?
        /// </summary>
        private bool runningScheduledJobs = false;

        private List<FFT_Job> jobs = new List<FFT_Job>();
        private List<JobHandle> fftJobHandles = new List<JobHandle>();

        [Range(0, 1)]
        private float calcFFTPorgress = 0;

        private bool fftJobRunning = false;

        #endregion Managed Data

        #region Managed Logic

        #region FFT Pre/Post Processing

        /// <summary>
        /// This is how far this FFT is done. Use this to monitor completion.
        /// </summary>
        public float Progress { get => calcFFTPorgress; private set => calcFFTPorgress = value; }

        /// <summary>
        /// Initiates the FFT spect data,
        /// </summary>
        private void InitFFTSpectData() {
            // Get the basic size data for this FFT.
            GetBasicFFTSizeInfo(out _, out int bucketsWidth, out int iterations);

            // Create ouput structures for the FFT data in case we're saving it in full blocks.
            usingSpectrumDataAssembler = true;
            spectrumData = new FFTSpectrumDataAssembler(iterations, bucketsWidth);
            if ( mySettings.AltOutput != FFTSharedSettings.AlternateOutputDataOption.None ) {
                spectrumDataDBV = new FFTSpectrumDataAssembler(iterations, bucketsWidth);
            }
        }

        /// <summary>
        /// Do the pre-processing work needed to set-up the FFT process -- this includes generating the size data, fft windowing data, then launching the FFT.
        /// </summary>
        private async UniTask FFTPreProcessing() {
            // Get the basic size data for this FFT.
            GetBasicFFTSizeInfo(out int sampleBlockSize, out int bucketsWidth, out int iterations);

            // Pre calculate the expensive to generate and static data used for windowing the FFT.
            double[] windowCoefs;
            double windowScaleFactor;
            if ( mySettings.UseJobsFFT ) {
                windowCoefs = null;
                windowScaleFactor = 0;
            } else {
                windowCoefs = DSP.Window.Coefficients(mySettings.FFTWindowFilter, (uint)sampleBlockSize);
                windowScaleFactor = DSP.Window.ScaleFactor.Signal(windowCoefs);
            }

            // Set if this script is going to be streamlined or not.
            IsStreaming = Settings.aam.UseStreamline;

            // Start Main Processing
            await ChooseFFT_ExecutionMethod(sampleBlockSize, bucketsWidth, iterations, windowCoefs, windowScaleFactor);
        }

        /// <summary>
        /// Generates the needed sizing information for this FFT from FFTSettings.
        /// </summary>
        /// <param name="sampleBlockSize"></param>
        /// <param name="bucketsWidth"></param>
        /// <param name="iterations">Number of FFT buckets that need to be </param>
        private void GetBasicFFTSizeInfo(out int sampleBlockSize, out int bucketsWidth, out int iterations) {
            // Get intial size info for the input sample and output block size from the FFTSettings.
            sampleBlockSize = mySettings.FFTSampleSize;
            bucketsWidth = mySettings.FFTBucketSize;
            iterations = mySettings.FFTMaxWindow;

            /* // This is the original code to get the FFTSize but it's just as fast to get it from the settings...
			// Check if the output data is using overlapping windows, as this doubles the size of the output data.
			int effectiveBlockSize = mySettings.UseOverlappingWindows ? sampleBlockSize / 2 : sampleBlockSize;

			// Generate needed size based off of the input parent length data.
			if ( mySettings.UseJobsFFT ) {
				iterations = ParentDataNA.Length / effectiveBlockSize;
			} else {
				iterations = ParentData.Length / effectiveBlockSize;
			}

			// Prevent the last window of the FFT from going out of range.
			// (This prevents the last window being is shifted half of the FFT's size beyond the max index...)
            if ( mySettings.UseOverlappingWindows ) iterations--;
			*/
        }

        /// <summary>
        /// Chooses which of the four internal methods are used to calculate the FFT.
        /// </summary>
        private async UniTask ChooseFFT_ExecutionMethod(int sampleBlockSize, int bucketsWidth, int iterations, double[] windowCoefs, double windowScaleFactor) {
            // Mark job started running
            fftJobRunning = true;

            UniTask task;
            // Start the job running one of serveral ways on a new thread
            if ( mySettings.UseJobsFFT ) {
                if ( mySettings.FFTJobs == 1 ) {
                    // Run the simple code which only sets up and runs one job (back-up clean working code).
                    task = SimpleSingleJobBasedFFT(iterations, sampleBlockSize, bucketsWidth);
                } else {
                    // Slightly more complex code which splits the code into multiple jobs.
                    task = SplitJobBasedFFT(iterations, sampleBlockSize, bucketsWidth);
                }
            } else { // Do the work single threaded using c# instead of the job's system.
                // Make the fft
                DSPLib.FFT fft = new DSPLib.FFT();
                fft.Initialize((uint)sampleBlockSize);

                // Choose execution method based upon the amount of debugging which is selected.
                if ( mySettings.aam.LogFFTAndBelowDetails.Use ) {
                    // Provides much much more detailed logs but has significant logging overhead.
                    task = FFTDetailedDiagnostic(iterations, sampleBlockSize, windowCoefs, windowScaleFactor, fft, bucketsWidth);
                } else {
                    // Slightly stramlined, small amount of logging used.
                    task = FFTSparseDiagnostic(iterations, sampleBlockSize, windowCoefs, windowScaleFactor, fft, bucketsWidth);
                }
            }

            // Wait for the task to be done, if we're just scheduling the
            await task;
            Progress = 1f;
            fftJobRunning = false;

            // Clear all links to parent data.
            ClearAllParentDataLinksOrCopies();

            AnalysisNodeStates newState;
            if ( usingSpectrumDataAssembler ) {
                // Verify that the data was processed correctly
                if ( spectrumData.AllBlocksLoaded ) {
                    newState = AnalysisNodeStates.OutputDataIdle;
                } else {
                    Debug.LogError(string.Format("Not all blocks were loaded in for the FFTJob [<{0}>].", mySettings.MyName));
                    newState = AnalysisNodeStates.InternalFailedValidation;
                }
            } else {
                newState = AnalysisNodeStates.OutputDataIdle;
            }

            // Notify job is completed.
            NotifySettingsJobScheduledOrEnded(newState);

            /*
			// Copy ouput data to native arrays for faster child reading...
			if ( mySettings.UseJobsFFT ) {
				this.StartCoroutineAsync(CopyDataToOuputNativeArrays(), out fftJobProcess);
				yield return fftJobProcess.Wait();
			}
			*/
        }

        /*
		/// <summary>
		/// Copies the data to output native arrays off of the main thread.
		/// </summary>
		private IEnumerator CopyDataToOuputNativeArrays() {
			managingSpectData_NA = true;
			spectDataNA = NArrayCopy.ToNativeArray(spectrumData.SpectData);
			spectrumData = null; // Reset the data assembling class to dealloc it's ram.

			if ( mySettings.CalculateDBV != FFTSharedSettings.DBVData.None ) {
				managingSpectDataDVB_NA = true;
				spectDataDBV = NArrayCopy.ToNativeArray(spectrumDataDBV.SpectData);
				spectrumDataDBV = null;// Reset the data assembling class to dealloc it's ram.
			}

			yield break;
		}
		*/

        #endregion FFT Pre/Post Processing

        #region C# FFT Processing Methods

        /// <summary>
        /// Performs the FFT analysis one a single thread with minmal or no diagnostic feedback.
        /// </summary>
        private async UniTask FFTSparseDiagnostic(int iterations, int sampleBlockSize,
                                             double[] windowCoefs, double windowScaleFactor, DSPLib.FFT fft, int bucketsWidth) {
            await UniTask.SwitchToThreadPool();

            TimingStart("C#");

            InitFFTSpectData();

            double[] sampleChunk = new double[sampleBlockSize];
            double[] scaledSpectrumChunk = new double[sampleBlockSize];
            Complex[] fftSpectrum = new Complex[bucketsWidth];
            double[] scaledFFTSpectrum = new double[bucketsWidth];
            double[] fftDBV = new double[bucketsWidth];

            int sampleBlockOffset = mySettings.OverlappingWindows > 0 ? sampleBlockSize / (int)Mathf.Pow(2, mySettings.OverlappingWindows) : sampleBlockSize;

            Debug.LogFormat("Offset: {0}, Iterations: {1}, ParentData/Offset: {2}", sampleBlockOffset, iterations, ParentData.Length / sampleBlockOffset);

            // Do the main loop
            if ( mySettings.AltOutput == FFTSharedSettings.AlternateOutputDataOption.None ) {
                for ( int i = 0; i < iterations; i++ ) {
                    // Grab the current (analysisSetup.FFTSamples) chunk of audio sample data
                    Array.Copy(ParentData, i * sampleBlockOffset, sampleChunk, 0, sampleBlockSize);

                    // Apply our chosen FFT Window
                    scaledSpectrumChunk = DSP.Math.Multiply(sampleChunk, windowCoefs);

                    // Get the time it took to load the FFT data.
                    fftSpectrum = fft.Execute(scaledSpectrumChunk);

                    // Perform the FFT and convert output (complex numbers) to Magnitude
                    scaledFFTSpectrum = DSP.ConvertComplex.ToMagnitude(fftSpectrum);

                    // Even out the scale factors
                    scaledFFTSpectrum = DSP.Math.Multiply(scaledFFTSpectrum, windowScaleFactor);

                    // Save the progeess
                    spectrumData.SaveBlock(i, scaledFFTSpectrum);

                    // Update the progress bar
                    Progress = i / (float)iterations;
                }
            } else if ( mySettings.AltOutput == FFTSharedSettings.AlternateOutputDataOption.Scaled ) {
                for ( int i = 0; i < iterations; i++ ) {
                    // Grab the current (analysisSetup.FFTSamples) chunk of audio sample data
                    Array.Copy(ParentData, i * sampleBlockOffset, sampleChunk, 0, sampleBlockSize);

                    // Apply our chosen FFT Window
                    scaledSpectrumChunk = DSP.Math.Multiply(sampleChunk, windowCoefs);

                    // Get the time it took to load the FFT data.
                    fftSpectrum = fft.Execute(scaledSpectrumChunk);

                    // Perform the FFT and convert output (complex numbers) to Magnitude
                    scaledFFTSpectrum = DSP.ConvertComplex.ToMagnitude(fftSpectrum);

                    // Even out the scale factors, and convert to dBV
                    for ( int j = 0; j < bucketsWidth; j++ ) {
                        double re = fftSpectrum[j].Real;
                        double im = fftSpectrum[j].Imaginary;
                        fftDBV[j] = ( 2f * math.sqrt(re * re + im * im) ) * windowScaleFactor;
                    }

                    scaledFFTSpectrum = DSP.Math.Multiply(scaledFFTSpectrum, windowScaleFactor);

                    // Save the progeess
                    spectrumData.SaveBlock(i, scaledFFTSpectrum);
                    spectrumDataDBV.SaveBlock(i, fftDBV);

                    // Update the progress bar
                    Progress = i / (float)iterations;
                }
            } else {
                for ( int i = 0; i < iterations; i++ ) {
                    // Grab the current (analysisSetup.FFTSamples) chunk of audio sample data
                    Array.Copy(ParentData, i * sampleBlockOffset, sampleChunk, 0, sampleBlockSize);

                    // Apply our chosen FFT Window
                    scaledSpectrumChunk = DSP.Math.Multiply(sampleChunk, windowCoefs);

                    // Get the time it took to load the FFT data.
                    fftSpectrum = fft.Execute(scaledSpectrumChunk);

                    // Perform the FFT and convert output (complex numbers) to Magnitude
                    scaledFFTSpectrum = DSP.ConvertComplex.ToMagnitude(fftSpectrum);

                    // Even out the scale factors, and convert to dBV
                    MagnitudeToDBV(scaledFFTSpectrum, windowScaleFactor, ref fftDBV);
                    scaledFFTSpectrum = DSP.Math.Multiply(scaledFFTSpectrum, windowScaleFactor);

                    // Save the progeess
                    spectrumData.SaveBlock(i, scaledFFTSpectrum);
                    spectrumDataDBV.SaveBlock(i, fftDBV);

                    // Update the progress bar
                    Progress = i / (float)iterations;
                }
            }

            // Make the last window safe for overlapping windows....
            ZeroOutLastWindowForOverlappingWindows(bucketsWidth);

            TimingEnd(iterations);
        }

        /// <summary>
        /// Performs the FFT analysis one a single thread with heavy diagnostic feedback.
        /// </summary>
        private async UniTask FFTDetailedDiagnostic(int iterations, int sampleBlockSize,
                                                double[] windowCoefs, double windowScaleFactor, DSPLib.FFT fft, int bucketsWidth) {
            await UniTask.SwitchToThreadPool();

            TimingStart("High Detail Timing C#", true);

            // Init all all of the stopwatches.
            QuickWatch createArrays = new QuickWatch(string.Format("{0} - Create Arrays", LogName), false, false);
            QuickWatch arrayCopyTimer = new QuickWatch(string.Format("{0} - ArrayCpy", LogName), false, false);
            QuickWatch fftCoefApply = new QuickWatch(string.Format("{0} - Window Applying", LogName), false, false);
            QuickWatch fftTimer = new QuickWatch(string.Format("{0} - Analysis", LogName), false, false);
            QuickWatch fftToMag = new QuickWatch(string.Format("{0} - FFT To Mag", LogName), false, false);
            QuickWatch toSave = new QuickWatch(string.Format("{0} - Saving", LogName), false, false);

            createArrays.Start();
            InitFFTSpectData();

            double[] sampleChunk = new double[sampleBlockSize];
            double[] scaledSpectrumChunk = new double[sampleBlockSize];
            Complex[] fftSpectrum = new Complex[bucketsWidth];
            double[] scaledFFTSpectrum = new double[bucketsWidth];
            double[] fftDBV = new double[bucketsWidth];

            createArrays.Stop();

            int sampleBlockOffset = mySettings.OverlappingWindows > 0 ? sampleBlockSize / (int)Mathf.Pow(2, mySettings.OverlappingWindows) : sampleBlockSize;

            // Do the main loop
            if ( mySettings.AltOutput == FFTSharedSettings.AlternateOutputDataOption.None ) {
                for ( int i = 0; i < iterations; i++ ) {
                    // Grab the current (analysisSetup.FFTSamples) chunk of audio sample data
                    arrayCopyTimer.Start();
                    Array.Copy(ParentData, i * sampleBlockOffset, sampleChunk, 0, sampleBlockSize);
                    arrayCopyTimer.Stop();

                    // Apply our chosen FFT Window
                    fftCoefApply.Start();
                    scaledSpectrumChunk = DSP.Math.Multiply(sampleChunk, windowCoefs);
                    fftCoefApply.Stop();

                    // Get the time it took to load the FFT data.
                    fftTimer.Start();
                    fftSpectrum = fft.Execute(scaledSpectrumChunk);
                    fftTimer.Stop();

                    // Perform the FFT and convert output (complex numbers) to Magnitude
                    fftToMag.Start();
                    scaledFFTSpectrum = DSP.ConvertComplex.ToMagnitude(fftSpectrum);
                    scaledFFTSpectrum = DSP.Math.Multiply(scaledFFTSpectrum, windowScaleFactor);
                    fftToMag.Stop();

                    // Save the progeess
                    toSave.Start();
                    spectrumData.SaveBlock(i, scaledFFTSpectrum);
                    toSave.Stop();

                    // Update the progress bar
                    Progress = i / (float)iterations; ;
                }
            } else if ( mySettings.AltOutput == FFTSharedSettings.AlternateOutputDataOption.Scaled ) {
                for ( int i = 0; i < iterations; i++ ) {
                    // Grab the current (analysisSetup.FFTSamples) chunk of audio sample data
                    arrayCopyTimer.Start();
                    Array.Copy(ParentData, i * sampleBlockOffset, sampleChunk, 0, sampleBlockSize);
                    arrayCopyTimer.Stop();

                    // Apply our chosen FFT Window
                    fftCoefApply.Start();
                    scaledSpectrumChunk = DSP.Math.Multiply(sampleChunk, windowCoefs);
                    fftCoefApply.Stop();

                    // Get the time it took to load the FFT data.
                    fftTimer.Start();
                    fftSpectrum = fft.Execute(scaledSpectrumChunk);
                    fftTimer.Stop();

                    // Perform the FFT and convert output (complex numbers) to Magnitude
                    fftToMag.Start();
                    scaledFFTSpectrum = DSP.ConvertComplex.ToMagnitude(fftSpectrum);
                    for ( int j = 0; j < bucketsWidth; j++ ) {
                        double re = fftSpectrum[j].Real;
                        double im = fftSpectrum[j].Imaginary;
                        fftDBV[j] = ( 2f * math.sqrt(re * re + im * im) ) * windowScaleFactor;
                    }
                    scaledFFTSpectrum = DSP.Math.Multiply(scaledFFTSpectrum, windowScaleFactor);
                    fftToMag.Stop();

                    // Save the progeess
                    toSave.Start();
                    spectrumData.SaveBlock(i, scaledFFTSpectrum);
                    spectrumDataDBV.SaveBlock(i, fftDBV);
                    toSave.Stop();

                    // Update the progress bar
                    Progress = i / (float)iterations; ;
                }
            } else {
                for ( int i = 0; i < iterations; i++ ) {
                    // Grab the current (analysisSetup.FFTSamples) chunk of audio sample data
                    arrayCopyTimer.Start();
                    Array.Copy(ParentData, i * sampleBlockOffset, sampleChunk, 0, sampleBlockSize);
                    arrayCopyTimer.Stop();

                    // Apply our chosen FFT Window
                    fftCoefApply.Start();
                    scaledSpectrumChunk = DSP.Math.Multiply(sampleChunk, windowCoefs);
                    fftCoefApply.Stop();

                    // Get the time it took to load the FFT data.
                    fftTimer.Start();
                    fftSpectrum = fft.Execute(scaledSpectrumChunk);
                    fftTimer.Stop();

                    // Perform the FFT and convert output (complex numbers) to Magnitude
                    fftToMag.Start();
                    scaledFFTSpectrum = DSP.ConvertComplex.ToMagnitude(fftSpectrum);
                    MagnitudeToDBV(scaledFFTSpectrum, windowScaleFactor, ref fftDBV);
                    scaledFFTSpectrum = DSP.Math.Multiply(scaledFFTSpectrum, windowScaleFactor);
                    fftToMag.Stop();

                    // Save the progeess
                    toSave.Start();
                    spectrumData.SaveBlock(i, scaledFFTSpectrum);
                    spectrumDataDBV.SaveBlock(i, fftDBV);
                    toSave.Stop();

                    // Update the progress bar
                    Progress = i / (float)iterations; ;
                }
            }

            // Make the last window safe for overlapping windows....
            ZeroOutLastWindowForOverlappingWindows(bucketsWidth);

            // Log the analysis time.
            arrayCopyTimer.StopAndReport(iterations);
            fftCoefApply.StopAndReport(iterations);
            fftTimer.StopAndReport(iterations);
            fftToMag.StopAndReport(iterations);
            toSave.StopAndReport(iterations);

            // Overall time
            TimingEnd(iterations);
        }

        /// <summary>
        /// Zeroes out the last windows of FFT data when using overlapping windows...
        /// </summary>
        private void ZeroOutLastWindowForOverlappingWindows(int bucketsWidth) {
            // Mark the last block of the FFTData as all zerod out if using overlapping windows
            if ( mySettings.OverlappingWindows > 0 ) {
                // When using overlapping windows the last bucket needs to have it's output data zeroed out,
                // because this bucket is not processed.

                int len = spectrumData.SpectData.Length - 1;
                for ( int j = len - bucketsWidth; j < len; j++ ) {
                    spectrumData.SpectData[j] = 0;
                }
                if ( mySettings.AltOutput != FFTSharedSettings.AlternateOutputDataOption.None ) {
                    for ( int j = len - bucketsWidth; j < len; j++ ) {
                        spectrumDataDBV.SpectData[j] = float.Epsilon;
                    }
                }
            }
        }

        /// <summary>
        /// Convert Complex DFT/FFT Result to: Log Magnitude dBV
        /// </summary>
        /// <param name="magFFT"> double[] input array"></param>
        /// <returns>double[] Magnitude Format (dBV)</returns>
        private static void MagnitudeToDBV(double[] magFFT, double windowScaleFactor, ref double[] outputDBV) {
            int length = magFFT.Length;
            for ( int i = 0; i < length; i++ ) {
                double magVal = magFFT[i];

                if ( magVal <= 0.0 )
                    magVal = double.Epsilon;

                outputDBV[i] = 20 * math.log10(magVal) * windowScaleFactor;
            }
        }

        #endregion C# FFT Processing Methods

        #region Job System FFT Processing Methods

        /// <summary>
        /// Creats a batch of jobs which perform the FFT on many threads.
        /// </summary>
        private async UniTask SplitJobBasedFFT(int iterations, int sampleBlockSize, int bucketsWidth) {
            // Stop job from running two at the same time
            if ( runningScheduledJobs ) {
                Debug.LogError("Already running a jobs based fft in <{0}>, cannot start another one.");
                return;
            }

            await UniTask.SwitchToThreadPool();

            runningScheduledJobs = true;

            // Create lists to manage jobs
            int jobCount = mySettings.FFTJobs;
            int batchMaxSize = Mathf.CeilToInt(iterations / (float)jobCount);
            int batch = 0; // Batch counter

            const Allocator persist = Allocator.Persistent;
            const NativeArrayOptions unint = NativeArrayOptions.UninitializedMemory;

            // Do job set-up on a seperate thread
            TimingStart(string.Format("Split Job ({0} Threads)", jobCount));

            bool logSubSteps = !mySettings.aam.LogFFTAndBelowDetails.Use;
            QuickWatch createTime = new QuickWatch(string.Format("{0} Split - Jobs Creation", LogName), logSubSteps);

            jobs.Clear();
            fftJobHandles.Clear();
            List<bool> jobNotDisposed = new();

            // Generate the window to use for this DFT.
            NativeArray<double> windowCoefsNA = new NativeArray<double>(sampleBlockSize, Allocator.Persistent, unint);
            NativeArray<double>  windowCoefsSignalNA = new NativeArray<double>(1, Allocator.Persistent, unint);

            RamPeak.RecordNASize(windowCoefsNA, 1, false);
            RamPeak.RecordNASize(windowCoefsSignalNA, 1, false);

            GenerateWindowJob windowJob = new GenerateWindowJob() {
                Window = windowCoefsNA,
                WindowCoefsSignal = windowCoefsSignalNA,
                Z = new NativeArray<double>(sampleBlockSize, Allocator.Persistent, unint),
                WindowSize = Convert.ToUInt32(sampleBlockSize),
                UseWindow = mySettings.FFTWindowFilter,
            };

            // Create output data
            managingSpectData_NA = true;
            spectDataNA = new NativeArray<float>(bucketsWidth * iterations, Allocator.Persistent);
            managingSpectDataDVB_NA = true;
            int DBV_Size = mySettings.AltOutput == FFTSharedSettings.AlternateOutputDataOption.None ? 0 : bucketsWidth * iterations;
            spectDataDBV_NA = new NativeArray<float>(DBV_Size, Allocator.Persistent);

            RamPeak.RecordNASize(spectDataNA);
            RamPeak.RecordNASize(spectDataDBV_NA);

            OutputPeak.RecordNASize(spectDataNA);
            if ( DBV_Size > 0 ) OutputPeak.RecordNASize(spectDataDBV_NA);

            // i  = The starting block for this batch.
            for ( int i = 0; i < iterations - 1; i += batchMaxSize ) {
                // Calculate number of blocks used here
                int thisJobSize = ( iterations - i < batchMaxSize ) ? iterations - i : batchMaxSize;

                // Record Ram Use
                NativeArray<double> nativeArray = new NativeArray<double>(sampleBlockSize, persist, unint); // x4
                NativeArray<Complex> complices = new NativeArray<Complex>(bucketsWidth, persist, unint);  // x1
                NativeArray<double> nativeArray1 = new NativeArray<double>(bucketsWidth, persist, unint); // x3
                NativeArray<uint> nativeArray2 = new NativeArray<uint>(sampleBlockSize, persist, unint); // x1
                NativeArray<Complex> complices1 = new NativeArray<Complex>(sampleBlockSize, persist, unint); // x1

                RamPeak.RecordNASize(nativeArray, 4, false);
                RamPeak.RecordNASize(complices, 1, false);
                RamPeak.RecordNASize(nativeArray1, 3, false);
                RamPeak.RecordNASize(nativeArray2, 1, false);
                RamPeak.RecordNASize(complices1, 1, false);

                FFT_Job job = new FFT_Job {
                    // Input Data
                    AudioSamples = ParentDataNA,
                    WindowCoefs = windowCoefsNA,
                    WindowScaleFactor = windowCoefsSignalNA,
                    SampleBlockSize = sampleBlockSize,
                    SampleStepSize = mySettings.FFTStepSize,
                    BucketsWidth = bucketsWidth,
                    NumberOfBlocksJobProcesses = thisJobSize,
                    StartingBlock = i,
                    DBVMethod = mySettings.AltOutput,

                    // Job Working Arrays
                    SampleChunk = nativeArray,
                    ScaledSpectrumChunk = new NativeArray<double>(sampleBlockSize, persist, unint),
                    fftSpectrum = complices,
                    ScaledFFTSpectrum = nativeArray1,
                    fftDBV = new NativeArray<double>(bucketsWidth, persist, unint),
                    BlockOutput = new NativeArray<double>(bucketsWidth, persist, unint),

                    // FFT Working arrays
                    re = new NativeArray<double>(sampleBlockSize, persist, unint),
                    im = new NativeArray<double>(sampleBlockSize, persist, unint),
                    revTgt = nativeArray2,
                    unswizzle = complices1,

                    // Ouput data
                    outData = spectDataNA,
                    outDataDBv = spectDataDBV_NA,
                };

                // Add job to jobs list
                jobs.Add(job);
                batch++;
            }

            // When testing the number of output batches can be less than the fully used amount.
            int batchesMade = batch;

            createTime.StopAndReport();

            // Schedule job to run on main thread
            await UniTask.SwitchToMainThread();

            // Start jobs running from the main thread.
            QuickWatch jobsScheduling = new QuickWatch(string.Format("{0} Split - Jobs Scheduling", LogName), logSubSteps, true);

            // Get the streamlined parents job if we have dependents for the first job.
            JobHandle windowJobHandle = windowJob.Schedule();

            // Inject the parent's jobs as a dependency with the second [big] job so all of the other cores don't have to take a single thread break.
            if ( mySettings.HasAnyStreamlinedParents ) {
                windowJobHandle = JobHandle.CombineDependencies(windowJobHandle, mySettings.CombineParentsHandles());
            }

            for ( int i = 0; i < batchesMade; i++ ) {
                fftJobHandles.Add(jobs[i].Schedule(windowJobHandle));
                jobNotDisposed.Add(true);
            }

            // Combine all of the handles into a single one.
            JobHandle semiFinal = fftJobHandles.CombineAllHandlesThenClear();

            // Schedule the destruction of both of the working arrays.
            FinalHandle = JobHandle.CombineDependencies(windowCoefsNA.Dispose(semiFinal), windowCoefsSignalNA.Dispose(semiFinal));

            // Send along the final handle in case it's needed for deallocation by that parent in streamline mode.
            mySettings.SendAlongJobHandleToParents(FinalHandle);

            jobsScheduling.StopAndReport(logSubSteps, batchesMade);

            AudioAnalysisManager.MarkJobScheduledInAnalysisTree();

            // If this is not a streaming job exit here.
            if ( !IsStreaming ) await CompleteWhenDone();
            else runningScheduledJobs = false;
        }

        private async UniTask CompleteWhenDone() {
            // As jobs become done load them into the final results, off of the main thread.
            bool logSubSteps = !mySettings.aam.LogFFTAndBelowDetails.Use;
            QuickWatch jobsRunning = new QuickWatch(string.Format("{0} Split - Jobs Running", LogName), logSubSteps, true);

            while ( !FinalHandle.IsCompleted && runningScheduledJobs ) await UniTask.NextFrame();
            FinalHandle.Complete();

            jobsRunning.StopAndReport();

            CleanUpPostJobRun();
        }

        private void CleanUpPostJobRun() {
            // Release this job's monolith hold if it's using one.
            if ( mySettings.IsMonolithicTask && !IsStreaming ) {
                mySettings.aam.MonolithicJobSignalEarlyRelease(mySettings, FinalHandle);
            }

            // Dispose of all final data
            bool logSubSteps = !mySettings.aam.LogFFTAndBelowDetails.Use;
            QuickWatch naDisposal = new(string.Format("{0} - Disposeal", LogName), logSubSteps, true);

            // Dispose data immediatly (super small data outputs can be cleared quickly)
            Progress = 1f;

            naDisposal.StopAndReport();

            TimingEnd();
            runningScheduledJobs = false;
        }

        /// <summary>
        /// Creates a batch of jobs which perform the FFT on a single thread.
        /// </summary>
        private async UniTask SimpleSingleJobBasedFFT(int iterations, int sampleBlockSize, int bucketsWidth) {
            // Stop job from running two at the same time
            if ( runningScheduledJobs ) {
                Debug.LogError("Already running a jobs based fft in <{0}>, cannot start another one.");
                return;
            }
            runningScheduledJobs = true;

            await UniTask.SwitchToThreadPool();

            TimingStart("FFT Single Thread Job");

            // Temp working data
            const Allocator jobAllocator = Allocator.Persistent;
            const NativeArrayOptions uninitializedMemory = NativeArrayOptions.UninitializedMemory;
            jobs.Clear();
            fftJobHandles.Clear();

            bool logSubSteps = !mySettings.aam.LogFFTAndBelowDetails.Use;
            QuickWatch createTime = new QuickWatch(string.Format("FFT Single [{0}] - Job Creation", mySettings.MyName), logSubSteps);

            // Calculate number of blocks used here
            int thisJobSize = iterations;

            // Generate the window to use for this DFT.
            NativeArray<double> windowCoefsNA = new NativeArray<double>(sampleBlockSize, Allocator.Persistent, uninitializedMemory);
            NativeArray<double> windowCoefsSignalNA = new NativeArray<double>(1, Allocator.Persistent);

            GenerateWindowJob windowJob = new GenerateWindowJob() {
                Window = windowCoefsNA,
                WindowCoefsSignal = windowCoefsSignalNA,
                Z = new NativeArray<double>(sampleBlockSize, Allocator.Persistent, uninitializedMemory),
                WindowSize = Convert.ToUInt32(sampleBlockSize),
                UseWindow = mySettings.FFTWindowFilter,
            };

            // Create output data
            managingSpectData_NA = true;
            spectDataNA = new NativeArray<float>(bucketsWidth * thisJobSize, Allocator.Persistent);
            managingSpectDataDVB_NA = true;
            spectDataDBV_NA = new NativeArray<float>(mySettings.AltOutput == FFTSharedSettings.AlternateOutputDataOption.None ? 0 : bucketsWidth * thisJobSize, Allocator.Persistent);

            FFT_Job job = new FFT_Job {
                // Input Data
                AudioSamples = ParentDataNA,
                WindowCoefs = windowCoefsNA,
                WindowScaleFactor = windowCoefsSignalNA,
                SampleBlockSize = sampleBlockSize,
                SampleStepSize = mySettings.FFTStepSize,
                BucketsWidth = bucketsWidth,
                NumberOfBlocksJobProcesses = thisJobSize,
                StartingBlock = 0,
                DBVMethod = mySettings.AltOutput,

                // Job Working Arrays
                SampleChunk = new NativeArray<double>(sampleBlockSize, jobAllocator, uninitializedMemory),
                ScaledSpectrumChunk = new NativeArray<double>(sampleBlockSize, jobAllocator, uninitializedMemory),
                fftSpectrum = new NativeArray<Complex>(bucketsWidth, jobAllocator, uninitializedMemory),
                ScaledFFTSpectrum = new NativeArray<double>(bucketsWidth, jobAllocator, uninitializedMemory),
                fftDBV = new NativeArray<double>(bucketsWidth, jobAllocator, uninitializedMemory),
                BlockOutput = new NativeArray<double>(bucketsWidth, jobAllocator, uninitializedMemory),

                // FFT Working arrays
                re = new NativeArray<double>(sampleBlockSize, jobAllocator, uninitializedMemory),
                im = new NativeArray<double>(sampleBlockSize, jobAllocator, uninitializedMemory),
                revTgt = new NativeArray<uint>(sampleBlockSize, jobAllocator, uninitializedMemory),
                unswizzle = new NativeArray<Complex>(sampleBlockSize, jobAllocator, uninitializedMemory),

                // Ouput data
                outData = spectDataNA,
                outDataDBv = spectDataDBV_NA,
            };

            jobs.Add(job);

            createTime.StopAndReport();

            // Set job to run on main threads
            await UniTask.SwitchToMainThread();

            // Start the job
            // Get the streamlined parents job if we have dependents for the first job.
            JobHandle windowJobHandle = mySettings.HasAnyStreamlinedParents ? windowJob.Schedule(mySettings.CombineParentsHandles()) : windowJob.Schedule();

            JobHandle semiFinal = job.Schedule(windowJobHandle);

            FinalHandle = JobHandle.CombineDependencies(windowCoefsNA.Dispose(semiFinal), windowCoefsSignalNA.Dispose(semiFinal));

            fftJobHandles.Add(FinalHandle); // Add so that it can be completed if cleaning-up.

            AudioAnalysisManager.MarkJobScheduledInAnalysisTree();

            // If this is not a streaming job exit here.
            if ( !IsStreaming ) await CompleteWhenDone();
            else runningScheduledJobs = false;
        }

        #endregion Job System FFT Processing Methods

        #endregion Managed Logic

        #region IAnalysisJob

        [InlineEditor(Expanded = true)]
        [SerializeField]
        private FFTSettings mySettings;

        public override IAnalysisSettingsNode Settings {
            get => mySettings;
            protected set {
                if ( value is FFTSettings settings ) {
                    mySettings = settings;
                } else {
                    Debug.LogError(string.Format(
                        "The wrong type of Settings <{0}> was attempted to be set as a {1}'s settings.",
                        value.GetType().Name, this.GetType().Name));
                }
            }
        }

        public override bool OutputAvailable => calcFFTPorgress == 1f;

        public override bool NeedsCleanUp {
            get {
                // Check to see if the task is still running...
                if ( fftJobRunning ) return true;
                if ( spectrumData != null || spectrumDataDBV != null ) return true;
                if ( managingSpectDataDVB_NA || managingSpectData_NA ) return true;
                if ( calcFFTPorgress != 0 ) return true;
                if ( UsingAnyParentData ) return true;
                return false;
            }
        }

        public override bool CleanUp(bool clearNativeArraysNow = false) {
            // Check to see if the task is still running...
            if ( fftJobRunning ) {
                fftJobRunning = false;
            }

            calcFFTPorgress = 0;

            // Stop Jobs Based Jobs
            if ( runningScheduledJobs ) {
                runningScheduledJobs = false;

                // Complete whatever jobs are still running immediatly, and clear their managed data.
                for ( int i = 0; i < fftJobHandles.Count; i++ ) {
                    fftJobHandles[i].Complete();
                }

                // Clear Job Data
                fftJobHandles.Clear();
                jobs.Clear();
            }

            // Clear ouput data
            if ( mySettings.HasAnyStreamlinedChildren ) {
                JobHandle children = mySettings.CombineChildrenHandles();
                if ( managingSpectData_NA ) {
                    managingSpectData_NA = false;
                    StartCoroutine(ThreadDealloc.DeallocNativeArray(spectDataNA, children, clearNativeArraysNow));
                }
                if ( managingSpectDataDVB_NA ) {
                    managingSpectDataDVB_NA = false;
                    StartCoroutine(ThreadDealloc.DeallocNativeArray(spectDataDBV_NA, children, clearNativeArraysNow));
                }
            } else {
                if ( managingSpectData_NA ) {
                    managingSpectData_NA = false;
                    StartCoroutine(ThreadDealloc.DeallocNativeArray(spectDataNA, clearNativeArraysNow));
                }
                if ( managingSpectDataDVB_NA ) {
                    managingSpectDataDVB_NA = false;
                    StartCoroutine(ThreadDealloc.DeallocNativeArray(spectDataDBV_NA, clearNativeArraysNow));
                }
            }

            spectrumData = null;
            spectrumDataDBV = null;
            usingSpectrumDataAssembler = false;

            // Clear parent data
            if ( UsingAnyParentData )
                ClearAllParentDataLinksOrCopies();

            return true;
        }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously

        public override async UniTask<bool> StartJob() {
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
            if ( !fftJobRunning ) {
                // Set state to data processing
                mySettings.State = AnalysisNodeStates.RunningJob;

                // Get parent data depending on job type.
                if ( mySettings.UseJobsFFT ) {
                    SingleParentDataLink = GetParentDataNA(ref ParentDataNA, mySettings.MyParent);
                } else {
                    SingleParentDataLink = GetParentData(ref ParentData, mySettings.MyParent);
                }

                // Process data if parent data was recieved.
                if ( SingleParentDataLink != ParentDataLink.Unlinked ) {
                    // Mark this job as running
                    Settings.aam.MarkJobAsRunning(mySettings);

                    // Run the job
                    await FFTPreProcessing();
                    return true;
                }
            }
            return false;
        }

        protected override bool DoLogTime => Settings.aam.LogFFTAndBelow.Use;

        public override long CurrManagedDataSize {
            get {
                long sum = 0;
                if ( OutputAvailable ) {
                    if ( mySettings.UseJobsFFT ) {
                        if ( managingSpectData_NA ) sum += NativeDataSumViewer.GetSize_NA(spectDataNA);
                        if ( managingSpectDataDVB_NA ) sum += NativeDataSumViewer.GetSize_NA(spectDataDBV_NA);
                    } else {
                        if ( spectrumData != null ) sum += NativeDataSumViewer.GetSize(spectrumData.SpectData);
                        if ( spectrumDataDBV != null ) sum += NativeDataSumViewer.GetSize(spectrumData.SpectData);
                    }
                }
                return sum;
            }
        }

        /// <summary>
        /// Returns the data from the FFT
        /// </summary>
        /// <param name="option">Option 0 returns the default FFT data, anything else returns normalized FFT data used for DB conversion.</param>
        public override float[] GetData(int option = 0) {
            if ( option == 0 ) {
                return spectrumData.SpectData;
            } else {
                return spectrumDataDBV.SpectData;
            }
        }

        /// <summary>
        /// Returns the data from the FFT
        /// </summary>
        /// <param name="option">Option 0 returns the default FFT data, anything else returns normalized FFT data used for DB conversion.</param>
        public override NativeArray<float> GetDataNArray(int option = 0) {
            if ( option == 0 ) {
                return spectDataNA;
            } else {
                return spectDataDBV_NA;
            }
        }

        /// <summary>
        /// Not supported for this node type. Will throw error.
        /// </summary>
        public override NativeList<float> GetDataNList(int option = 0) {
            throw new NotImplementedException();
        }

        #endregion IAnalysisJob
    }
}