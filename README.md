# Burst-Saftey-Slowdown
Repository with a couple burst optimized DSP jobs. Test Results For Unity 2021.3.2f1 vs 2022.1.0f1 -- Showing the slow down of in-editor burst compiled code in 2022.1 when safety checks are turned on.

I've included two sample jobs (source code) with vastly different complexity levels as examples. For both of them I've added timing screen shots of the profiler showing them running the same track as above with and without safety checks for 2022.1.0f1. I also added a single test of the same code and track running in 2021.3.2f1 with safety checks for reference. I also included the full LLVM Optimized output, with and without safety checks, for both versions.

The simple job only contains around 9 operators -- mostly read / write. All it does is split a stereo wav file into three audio streams (left, mono, right). It's only showing around a 75% slow down with safety (on) between the two versions of the editor.

The other job is a full implementation of a FFT (around 500 lines of uncompiled code w/ comments). It's showing over a 10x slowdown between the two editor versions in the included tests (result). I included the original C# version I adapted it from for clarity.

I can't provide a full working unity project with these as that would require me putting too much of my game's core IP online, but I did include the full source for these two job as their algorithms are easily available everywhere; if not in this coding environment.
