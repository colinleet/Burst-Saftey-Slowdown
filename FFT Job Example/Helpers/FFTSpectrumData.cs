using System;
using System.Runtime.InteropServices;
using Unity.Collections;

namespace AudioAnalysis.Levels123 {

	[StructLayout(LayoutKind.Sequential)]
	public class FFTSpectrumDataAssembler {
		/// <summary>
		/// This is the size of the FFT frequency buckets.
		/// </summary>
		private int fftBucketSize;
		/// <summary>
		/// This is the count of how many windows are stored in thie FFTSpectrumData
		/// </summary>
		private int windowCounts;
		/// <summary>
		/// This is the number of blocks which have been saved.
		/// </summary>
		private int blocksDone = 0;
		/// <summary>
		/// This is a record of which block was saved.
		/// </summary>
		private bool[] blockSaved;
		/// <summary>
		/// This is the full block of all spectrum data.
		/// </summary>
		private float[] _spectData;

		/// <summary>
		/// Initializes a spectrum block and allows it to be saved.
		/// </summary>
		/// <param name="windowCounts"></param>
		/// <param name="fftBucketSize"></param>
		public FFTSpectrumDataAssembler(int windowCounts, int fftBucketSize) {
			this.windowCounts = windowCounts;
			this.fftBucketSize = fftBucketSize;
			_spectData = new float[WindowCounts * FFTBucketSize];

			// Save tracking data
			blockSaved = new bool[WindowCounts];
			for ( int i = 0; i < WindowCounts; i++ ) {
				BlockSaved[i] = false;
			}
			blocksDone = 0;
		}

		/// <summary>
		/// Returns the star location of a given memoery block.
		/// </summary>
		/// <param name="block"></param>
		/// <returns></returns>
		public int BlockLocation(int block) {
			return block * fftBucketSize;
		}

		/// <summary>
		/// Coppies a block to the local spectrum storage.
		/// </summary>
		/// <param name="block">Block copying in/</param>
		/// <param name="inSpectData">The spectrum data in double form.</param>
		public void SaveBlock(int block, double[] inSpectData) {
			if ( blockSaved[block] ) return;
            int destIndex = BlockLocation(block);
            int lengthToCopy = inSpectData.Length;
			for( int i = 0; i < lengthToCopy; i++ ) {
				_spectData[i] = (float)inSpectData[destIndex + i];

			}
			blockSaved[block] = true;
			blocksDone++;
		}

		/// <summary>
		/// Coppies a block to the local spectrum storage.
		/// </summary>
		/// <param name="block">Block copying in/</param>
		/// <param name="inSpectData">The spectrum data in double form.</param>
		public void SaveBlock(int block, NativeArray<double> inSpectData) {
			if ( blockSaved[block] ) return;
			int destIndex = BlockLocation(block);
			int lengthToCopy = inSpectData.Length;
			for ( int i = 0; i < lengthToCopy; i++ ) {
				_spectData[i] = (float)inSpectData[destIndex + i];
			}
			for ( int i = block; i < block + inSpectData.Length / FFTBucketSize; i++ ) {
				blockSaved[i] = true;
				blocksDone++;
			}
		}


		/// <summary>
		/// Returns if all of the blocks have been copied.
		/// </summary>
		public bool AllBlocksLoaded { get { return BlocksDone == WindowCounts; } }

		public int WindowCounts { get => windowCounts; }
		public int FFTBucketSize { get => fftBucketSize; }
		public bool[] BlockSaved { get => blockSaved;}
		public int BlocksDone { get => blocksDone; }
		public float[] SpectData { get => _spectData; internal set => _spectData = value; }
	}
}