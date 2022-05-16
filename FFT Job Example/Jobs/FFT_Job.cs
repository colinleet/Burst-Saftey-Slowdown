using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace AudioAnalysis.Nodes {

    /// <summary>
    /// This is a conversion of the FFT job which makes all of the logic which used to be managed by the FFT class to be managed internally instead using native arrays.
    /// </summary>
    [BurstCompile]
    internal struct FFT_Job : IJob {

        // Input Data
        [ReadOnly]
        public NativeArray<float> AudioSamples;

        [ReadOnly]
        public NativeArray<double> WindowCoefs;

        [ReadOnly]
        public NativeArray<double> WindowScaleFactor;

        /// <summary>
        /// This is the number of samples used for each FFT block.
        /// </summary>
        [ReadOnly]
        public int SampleBlockSize;

        /// <summary>
        /// This is how far each subsequent block is offset from the one previous to it.
        /// (For non-overlapping FFTs this is just SampleBlockSize, for overlapping its half / 3/4ths / 7/8ths)).
        /// </summary>
        [ReadOnly]
        public int SampleStepSize;

        [ReadOnly]
        public int BucketsWidth;

        [ReadOnly]
        public int NumberOfBlocksJobProcesses;

        [ReadOnly]
        public int StartingBlock;

        [ReadOnly]
        public Levels123.FFTSharedSettings.AlternateOutputDataOption DBVMethod;

        // Execute Working Data
        [DeallocateOnJobCompletion]
        public NativeArray<double> SampleChunk;

        [DeallocateOnJobCompletion]
        public NativeArray<double> ScaledSpectrumChunk;

        [DeallocateOnJobCompletion]
        public NativeArray<Complex> fftSpectrum;

        [DeallocateOnJobCompletion]
        public NativeArray<double> ScaledFFTSpectrum;

        [DeallocateOnJobCompletion]
        public NativeArray<double> fftDBV;

        [DeallocateOnJobCompletion]
        public NativeArray<double> BlockOutput;

        // FFT Working Data
        private double mFFTScale;

        private uint mLogN;           // log2 of FFT size
        private uint mN;              // Time series length
        private uint mLengthTotal;    // mN + mZp
        private uint mLengthHalf;     // (mN + mZp) / 2

        private const double sqrt2 = 1.4142135623730951f;       // sqrt(2)
        private const double sqrtHalf = 0.7071067811865476; // sqrt(.5) == 1 / sqrt(2)
        private const double neg2pi = -2 * math.PI_DBL;      // -2pi

        // FFTElement Data Unboxed From Class

        /// <summary>
        /// Holds the fft real values. This must be the length of mLengthTotal
        /// </summary>
        [DeallocateOnJobCompletion]
        public NativeArray<double> re;

        /// <summary>
        /// Holds the fft imaginary values. This must be the length of mLengthTotal
        /// </summary>
        [DeallocateOnJobCompletion]
        public NativeArray<double> im;

        [DeallocateOnJobCompletion]
        public NativeArray<UInt32> revTgt;

        [DeallocateOnJobCompletion]
        public NativeArray<Complex> unswizzle;

        // Output Data
        [NativeDisableContainerSafetyRestriction]
        [WriteOnly]
        public NativeArray<float> outData;

        [NativeDisableContainerSafetyRestriction]
        [WriteOnly]
        public NativeArray<float> outDataDBv;

        public void Execute() {
            // Initialize fft managed in local memory
            Initialize(Convert.ToUInt32(SampleBlockSize));

            //BucketsWidth
            int readingBlockStartIndex = SampleStepSize * StartingBlock;
            int writingBlockStartIndex = BucketsWidth * StartingBlock;

            // Dp the main calculation
            if ( DBVMethod == Levels123.FFTSharedSettings.AlternateOutputDataOption.None ) {
                CalcNoDBV(readingBlockStartIndex, writingBlockStartIndex);
            } else if ( DBVMethod == Levels123.FFTSharedSettings.AlternateOutputDataOption.Scaled ) {
                CalcScaledDVB(readingBlockStartIndex, writingBlockStartIndex);
            } else {
                CalcDBV(readingBlockStartIndex, writingBlockStartIndex);
            }
        }

        /// <summary>
        /// Copies the input into a temp working array.
        /// </summary>
        /// <param name="readingBlockStartIndex">Index to start reading from.</param>
        /// <param name="i">Index to copy from.</param>
        private void CopyInputFromFloatToDouble(int readingBlockStartIndex, int i) {
            int j = 0;
            int start = readingBlockStartIndex + i * SampleStepSize;
            int end = start + SampleBlockSize;
            for ( int k = start; k < end; k++ ) {
                SampleChunk[j++] = AudioSamples[k];
            }
        }

        /// <summary>
        /// Copies the input into a temp working array.
        /// </summary>
        /// <param name="readingBlockStartIndex">Index to start reading from.</param>
        /// <param name="i">Index to copy from.</param>
        private void CopyOutputDoubleToSinglePrimaryOutput(int writingBlockStartIndex, int i) {
            int j = 0;
            int writeStart = writingBlockStartIndex + i * BucketsWidth;
            int writeEnd = writeStart + BucketsWidth;
            for ( int k = writeStart; k < writeEnd; k++ ) {
                outData[k] = (float)BlockOutput[j++];
            }
        }

        /// <summary>
        /// Copies the input into a temp working array.
        /// </summary>
        /// <param name="readingBlockStartIndex">Index to start reading from.</param>
        /// <param name="i">Index to copy from.</param>
        private void CopyOutputDoubleToSingleSecondaryOutput(int writingBlockStartIndex, int i) {
            int j = 0;
            int writeStart = writingBlockStartIndex + i * BucketsWidth;
            int writeEnd = writeStart + BucketsWidth;
            for ( int k = writeStart; k < writeEnd; k++ ) {
                outDataDBv[k] = (float)BlockOutput[j++];
            }
        }

        private void CalcNoDBV(int readingBlockStartIndex, int writingBlockStartIndex) {
            for ( int i = 0; i < NumberOfBlocksJobProcesses; i++ ) {
                //if ( startingBlockIndex != 0 ) Debug.LogFormat("{0}/{1}", i, NumberOfBlocksJobProcesses);

                // Copy the new chunk of samples into the working array.
                CopyInputFromFloatToDouble(readingBlockStartIndex, i);

                // Perform the calculation
                // Apply our chosen FFT Window
                Multiply(SampleChunk, WindowCoefs, ref ScaledSpectrumChunk);

                // Get the time it took to load the FFT data.
                Execute(ScaledSpectrumChunk, ref fftSpectrum);

                // Perform the FFT and convert output (complex numbers) to magnitude, also convert to DB.
                ToMagnitude(fftSpectrum, ref ScaledFFTSpectrum);
                Multiply(ScaledFFTSpectrum, WindowScaleFactor[0], ref BlockOutput);

                // Copy to final array
                CopyOutputDoubleToSinglePrimaryOutput(writingBlockStartIndex, i);
            }
        }



        private void CalcScaledDVB(int readingBlockStartIndex, int writingBlockStartIndex) {
            double windowScaleFactor2 = WindowScaleFactor[0] * 2f;

            for ( int i = 0; i < NumberOfBlocksJobProcesses; i++ ) {
                // Copy the new chunk of samples into the working array.
                CopyInputFromFloatToDouble(readingBlockStartIndex, i);

                // Perform the calculation
                // Apply our chosen FFT Window
                Multiply(SampleChunk, WindowCoefs, ref ScaledSpectrumChunk);

                // Get the time it took to load the FFT data.
                Execute(ScaledSpectrumChunk, ref fftSpectrum);

                // Perform the FFT and convert output (complex numbers) to magnitude, also convert to DB.
                ToMagnitude(fftSpectrum, ref ScaledFFTSpectrum);

                for ( int j = 0; j < BucketsWidth; j++ ) {
                    double re = fftSpectrum[j].Real;
                    double im = fftSpectrum[j].Imaginary;
                    fftDBV[j] = math.sqrt(re * re + im * im) * windowScaleFactor2;
                }

                Multiply(ScaledFFTSpectrum, WindowScaleFactor[0], ref BlockOutput);

                // Copy to final array
                CopyOutputDoubleToSinglePrimaryOutput(writingBlockStartIndex, i);
                CopyOutputDoubleToSingleSecondaryOutput(writingBlockStartIndex, i);
            }
        }

        private void CalcDBV(int readingBlockStartIndex, int writingBlockStartIndex) {
            for ( int i = 0; i < NumberOfBlocksJobProcesses; i++ ) {
                // Copy the new chunk of samples into the working array.
                CopyInputFromFloatToDouble(readingBlockStartIndex, i);

                // Perform the calculation
                // Apply our chosen FFT Window
                Multiply(SampleChunk, WindowCoefs, ref ScaledSpectrumChunk);

                // Get the time it took to load the FFT data.
                Execute(ScaledSpectrumChunk, ref fftSpectrum);

                // Perform the FFT and convert output (complex numbers) to magnitude, also convert to DB.
                ToMagnitude(fftSpectrum, ref ScaledFFTSpectrum);
                MagnitudeToDBV(ScaledFFTSpectrum, WindowScaleFactor[0], ref fftDBV);
                Multiply(ScaledFFTSpectrum, WindowScaleFactor[0], ref BlockOutput);

                // Copy to final array
                CopyOutputDoubleToSinglePrimaryOutput(writingBlockStartIndex, i);
                CopyOutputDoubleToSingleSecondaryOutput(writingBlockStartIndex, i);
            }
        }

        #region Customized DSP. Math functions modified to work with native arrays.

        /// <summary>
        /// c[] = a[] * b[]
        /// </summary>
        private static void Multiply(NativeArray<double> a, NativeArray<double> b, ref NativeArray<double> c) {
            // Debug.Assert(( a.Length == b.Length ), "Length of arrays a[], b[] must match.");
            // UnityEngine.Debug.Log(string.Format("{0} {1} {2}", a.Length, b.Length, c.Length));
            int length = a.Length;
            for ( int i = 0; i < length; i++ ) {
                c[i] = a[i] * b[i];
            }
        }

        /// <summary>
        /// Convert Complex DFT/FFT Result to: Magnitude Vrms
        /// </summary>
        /// <param name="rawFFT"></param>
        /// <returns>double[] Magnitude Format (Vrms)</returns>
        private static void ToMagnitude(NativeArray<Complex> rawFFT, ref NativeArray<double> output) {
            int length = rawFFT.Length;
            for ( int i = 0; i < length; i++ ) {
                output[i] = rawFFT[i].Magnitude;
            }
        }

        /// <summary>
        /// Convert Complex DFT/FFT Result to: Log Magnitude dBV
        /// </summary>
        /// <param name="magFFT"> double[] input array"></param>
        /// <returns>double[] Magnitude Format (dBV)</returns>
        private static void MagnitudeToDBV(NativeArray<double> magFFT, double windowScaleFactor, ref NativeArray<double> outputDBV) {
            int length = magFFT.Length;
            for ( int i = 0; i < length; i++ ) {
                double magVal = magFFT[i];

                if ( magVal <= 0.0 )
                    magVal = double.Epsilon;

                outputDBV[i] = 20 * math.log10(magVal) * windowScaleFactor;
            }
        }

        /// <summary>
        /// c[] = a[] * b
        /// </summary>
        private static void Multiply(NativeArray<double> a, double b, ref NativeArray<double> c) {
            int length = a.Length;
            for ( int i = 0; i < length; i++ ) {
                c[i] = a[i] * b;
            }
        }

        #endregion Customized DSP. Math functions modified to work with native arrays.

        #region Unclassed FFT

        #region FFT Core Functions

        /// <summary>
        /// Initialize the FFT. Must call first and this anytime the FFT setup changes.
        /// </summary>
        /// <param name="inputDataLength"></param>
        /// <param name="zeroPaddingLength"></param>
        private void Initialize(uint inputDataLength, uint zeroPaddingLength = 0) {
            // Set variables which were intialized by the class
            mFFTScale = 1.0;
            mLogN = 0;       // log2 of FFT size
            mN = 0;          // Time series length

            // Start original function
            mN = inputDataLength;

            // Find the power of two for the total FFT size up to 2^32
            //bool foundIt = false;
            for ( mLogN = 1; mLogN <= 32; mLogN++ ) {
                double n = Math.Pow(2.0, mLogN);
                if ( ( inputDataLength + zeroPaddingLength ) == n ) {
                    //foundIt = true;
                    break;
                }
            }

            //if ( foundIt == false )
            //    throw new ArgumentOutOfRangeException("inputDataLength + zeroPaddingLength was not an even power of 2! FFT cannot continue.");

            // Set global parameters.
            mLengthTotal = inputDataLength + zeroPaddingLength;
            mLengthHalf = ( mLengthTotal / 2 ) + 1;

            // Set the overall scale factor for all the terms
            mFFTScale = sqrt2 / mLengthTotal;                // Natural FFT Scale Factor  // Window Scale Factor
            mFFTScale *= ( (double)mLengthTotal ) / inputDataLength;    // Zero Padding Scale Factor

            // Specify target for bit reversal re-ordering.
            int kI = 0;
            for ( uint k = 0; k < ( mLengthTotal ); k++ ) {
                revTgt[kI] = BitReverse(k, mLogN);
                kI++;
            }
        }

        /// <summary>
        /// Executes a FFT of the input time series.
        /// </summary>
        /// <param name="timeSeries"></param>
        private void Execute(NativeArray<double> timeSeries, ref NativeArray<Complex> results) {
            uint numFlies = mLengthTotal >> 1;  // Number of butterflies per sub-FFT
            uint span = mLengthTotal >> 1;      // Width of the butterfly
            uint spacing = mLengthTotal;        // Distance between start of sub-FFTs
            uint wIndexStep = 1;          // Increment for twiddle table index

            //Debug.Assert(timeSeries.Length <= mLengthTotal, "The input timeSeries length was greater than the total number of points that was initialized. FFT.Exectue()");

            // Copy data into linked complex number objects
            int k = 0;
            for ( uint i = 0; i < mN; i++ ) {
                re[k] = timeSeries[k];
                im[k] = 0.0;
                k++;
            }

            // If zero padded, clean the 2nd half of the linked list from previous results
            if ( mN != mLengthTotal ) {
                k = Convert.ToInt32(mN);
                for ( uint i = mN; i < mLengthTotal; i++ ) {
                    re[k] = 0.0;
                    im[k] = 0.0;
                    k++;
                }
            }

            // For each stage of the FFT
            for ( uint stage = 0; stage < mLogN; stage++ ) {
                // Compute a multiplier factor for the "twiddle factors".
                // The twiddle factors are complex unit vectors spaced at
                // regular angular intervals. The angle by which the twiddle
                // factor advances depends on the FFT stage. In many FFT
                // implementations the twiddle factors are cached, but because
                // array lookup is relatively slow in C#, it's just
                // as fast to compute them on the fly.
                double wAngleInc = wIndexStep * neg2pi / mLengthTotal;
                double wMulRe = math.cos(wAngleInc);
                double wMulIm = math.sin(wAngleInc);

                for ( uint start = 0; start < ( mLengthTotal ); start += spacing ) {
                    int iXTop = Convert.ToInt32(start);
                    int iXBot = Convert.ToInt32(start + span);

                    double wRe = 1.0;
                    double wIm = 0.0;

                    // For each butterfly in this stage
                    for ( uint flyCount = 0; flyCount < numFlies; ++flyCount ) {
                        // Get the top & bottom values
                        double xTopRe = re[iXTop];
                        double xTopIm = im[iXTop];
                        double xBotRe = re[iXBot];
                        double xBotIm = im[iXBot];

                        // Top branch of butterfly has addition
                        re[iXTop] = xTopRe + xBotRe;
                        im[iXTop] = xTopIm + xBotIm;

                        // Bottom branch of butterfly has subtraction,
                        // followed by multiplication by twiddle factor
                        xBotRe = xTopRe - xBotRe;
                        xBotIm = xTopIm - xBotIm;
                        re[iXBot] = xBotRe * wRe - xBotIm * wIm;
                        im[iXBot] = xBotRe * wIm + xBotIm * wRe;

                        // Advance butterfly to next top & bottom positions
                        iXTop++;
                        iXBot++;

                        // Update the twiddle factor, via complex multiply
                        // by unit vector with the appropriate angle
                        // (wRe + j wIm) = (wRe + j wIm) x (wMulRe + j wMulIm)
                        double tRe = wRe;
                        wRe = wRe * wMulRe - wIm * wMulIm;
                        wIm = tRe * wMulIm + wIm * wMulRe;
                    }
                }

                numFlies >>= 1;   // Divide by 2 by right shift
                span >>= 1;
                spacing >>= 1;
                wIndexStep <<= 1;     // Multiply by 2 by left shift
            }

            // The algorithm leaves the result in a scrambled order.
            // Unscramble while copying values from the complex
            // linked list elements to a complex output vector & properly apply scale factors.
            int length = revTgt.Length;
            for ( int i = 0; i < length; i++ ) {
                unswizzle[(int)revTgt[i]] = new Complex(re[i] * mFFTScale, im[i] * mFFTScale);
            }

            // Return 1/2 the FFT result from DC to Fs/2 (The real part of the spectrum)
            //UInt32 halfLength = ((mN + mZp) / 2) + 1;

            // Complex[] result = new Complex[mLengthHalf];
            //Array.Copy(unswizzle, result, mLengthHalf);
            NativeArray<Complex>.Copy(unswizzle, results, (int)mLengthHalf);

            // DC and Fs/2 Points are scaled differently, since they have only a real part
            results[0] = new Complex(results[0].Real * sqrtHalf, 0.0);
            results[(int)mLengthHalf - 1] = new Complex(results[(int)mLengthHalf - 1].Real * sqrtHalf, 0.0);
        }

        #endregion FFT Core Functions

        #region Private FFT Routines

        //
        // * Do bit reversal of specified number of places of an int
        // * For example, 1101 bit-reversed is 1011
        // *
        // * @param   x       Number to be bit-reverse.
        // * @param   numBits Number of bits in the number.
        //

        private static uint BitReverse(uint x, uint numBits) {
            uint y = 0;
            for ( uint i = 0; i < numBits; i++ ) {
                y <<= 1;
                y |= x & 0x0001;
                x >>= 1;
            }
            return y;
        }

        #endregion Private FFT Routines

        #endregion Unclassed FFT
    }
}