.. _global_referenced_desc:

Global Referenced
=================

Global referenced temporal coding algorithms rely on the alignment of spike generation with a global temporal characteristic of the input signal. Two prominent methods in this category are:

- **Phase Encoding**: This method utilizes the time difference between the input signal and an oscillatory reference to generate spikes. The oscillatory reference serves as a global temporal marker, ensuring that the spikes encode information about the timing relative to this oscillation. This technique is particularly effective in scenarios where the temporal structure of the input signal is critical, such as in phase-locked neural activities (Hopfield, 1995).

- **Time-to-First-Spike (TTFS)**: Unlike Phase Encoding, TTFS relies on the time elapsed since the onset of a stimulus to generate spikes. The first spike's timing carries information about the stimulus's intensity or other relevant features. TTFS is widely used in systems where the speed of response is crucial, such as in sensory processing tasks (Thorpe and Gautrais, 1998; Johansson and Birznieks, 2004).

Similar to the Deconvolution-based algorithms, both Phase Encoding and TTFS produce spikes with single polarity, emphasizing the importance of temporal precision in neural coding.

.. toctree::
   :maxdepth: 1

   phase_encoding
   time_to_first_spike