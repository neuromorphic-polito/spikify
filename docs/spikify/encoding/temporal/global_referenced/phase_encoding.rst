.. _phase_encoding_algorithm_desc:

Phase Encoding
============================

The concept of encoding based on phase evaluation is rooted in the idea of using an oscillatory reference to encode information. This technique was explored by Montemurro et al. (2008), where the phase of an oscillatory reference is evaluated for encoding. In our implementation, we follow the approach proposed by Kim et al. (2018), where the binary representation of the input, using β fractional bits, serves as the oscillatory reference. The signal is first rectified and normalized for each channel into the range [0, 1].

**Algorithm Overview**:

The phase encoding method works as follows:

1. **Normalization**:
   The input signal is normalized within the range [0, 1] for each channel, ensuring the signal fits within the required bounds.

2. **Phase Calculation**:
   The normalized signal is then used to compute the phase angle, which is an essential step for phase encoding. The phase is calculated using the arcsin function, which effectively maps the normalized signal to a phase value.

3. **Quantization**:
   The computed phase values are then quantized based on the number of bits specified. The phase angle is divided into discrete levels, each representing a binary value.

4. **Binary Representation**:
   Finally, the quantized phase levels are converted into a binary representation. This binary sequence represents the encoded spikes.

**Implementation Steps**:

1. **Normalize the Signal**: Scale the signal to fit within the range [0, 1].
2. **Compute Phase Angles**: Use the arcsin function to determine the phase angles based on the normalized signal.
3. **Quantize the Phase**: Divide the phase values into discrete levels and assign them binary codes.
4. **Generate Spike Train**: Convert the quantized levels into a binary spike train.

**Advantages**:

Phase encoding is a powerful method that leverages phase information to represent the input signal. This technique is particularly effective for signals that are inherently oscillatory or when the phase information provides significant encoding advantages.

For a practical implementation in Python, see the :ref:`Phase Encoding Function <phase_encoding_function>`.

**References**:

- Montemurro, M. A., et al. (2008). "Phase-of-Firing Coding of Natural Scenes in Primary Visual Cortex." *Current Biology*.
- Kim, S., et al. (2018). "Binary Phase Encoding for Oscillatory Reference Signals." *Journal of Computational Neuroscience*.
