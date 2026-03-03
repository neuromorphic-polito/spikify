.. _phase_encoding_algorithm_desc:

Phase Encoding (PE)
============================

Phase Encoding (PE) is a temporal coding technique that encodes information by evaluating the phase angle of an input signal. This method leverages the relative timing of spikes to the phase of these rhythms to significantly boost the information carried by spike patterns and stabilize representations against noise.

In implementations like the one in this library, the signal is rectified, normalized, and the mean per block is mapped to a phase angle via arcsin. The phase is quantized into discrete levels (using β fractional bits as the oscillatory reference) and converted to a binary spike pattern, producing unipolar spikes.

**Algorithm Overview**

1. **Normalization**: Rectify and normalize the signal to [0, 1] per channel.

2. **Phase Calculation**: Compute phase as arcsin(mean) for block mean intensity.

3. **Quantization**: Map phase to discrete levels in [0, 2^num_bits - 1].

4. **Binary Representation**: Convert level to binary bits (right-shifted & masked), forming the spike pattern.

**Detailed Pseudocode**

.. code-block:: none
   :linenos:

   Phase Encoding Algorithm
   input: s signal (length T), num_bits (block size)

   out = zeros(T)

   n_blocks = T // num_bits
   block_means = mean(s over blocks of size num_bits)

   bins = linspace(0, π/2, 2^num_bits + 1)

   for block_idx = 0 to n_blocks-1
      mean = block_means[block_idx]
      phase = arcsin(mean)
      level = searchsorted(bins, phase)
      level = clip(level, 0, 2^num_bits - 1)
      bits = (level >> arange(num_bits-1, -1, -1)) & 1
      global_start = block_idx * num_bits
      out[global_start : global_start + num_bits] = bits
   end for

   output: out

**Advantages**

- Stabilizes representations against sensory noise by nesting spikes in low-frequency rhythms.
- Biologically plausible for sensory cortices.

**Disadvantages**

- Quantization limits precision; sensitive to normalization and bit depth.

For a practical implementation in Python, see the :ref:`Phase Encoding Function <phase_encoding_function>`.

**References**

- Montemurro, M. A., et al. (2008). "Phase-of-firing coding of natural visual stimuli in primary visual cortex." *Current Biology*.
- Kim, S., et al. (2018). "Deep neural networks with weighted spikes." *Neurocomputing*.
