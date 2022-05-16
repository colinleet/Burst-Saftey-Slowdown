using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace AudioAnalysis.Levels123 {

    /// <summary>
    /// This jobs converts sterio audio data into a left, right, and (sum) mono channel.
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct SterioToLMRSumJob : IJobParallelFor {

        /// <summary>
        /// All of the raw audio samples.
        /// </summary>
        [ReadOnly]
        public NativeArray<float> AudioSamples;

        /// <summary>
        /// Stream where the audio output will be placed.
        /// </summary>
        [WriteOnly]
        public NativeArray<float> Left;

        /// <summary>
        /// Stream where the audio output will be placed.
        /// </summary>
        [WriteOnly]
        public NativeArray<float> Mono;

        /// <summary>
        /// Stream where the audio output will be placed.
        /// </summary>
        [WriteOnly]
        public NativeArray<float> Right;

        public void Execute(int index) {
            // Sum two samples and put the ouput to the mono stream.
            int loc = index * 2;
            float left = AudioSamples[loc];
            float right = AudioSamples[loc + 1];
            float mono = left + right;
            Left[index] = left;
            Right[index] = right;
            Mono[index] = mono;
        }
    }
}

