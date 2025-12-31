.. _zero_cross_step_forward_algorithm_desc:

Zero-Crossing Step-Forward (ZCSF) Encoding
==============================================

The Zero-Crossing Step-Forward (ZCSF) encoding is a minimalist temporal contrast encoding technique that focuses exclusively on significant positive excursions of the signal. It applies half-wave rectification (setting all negative values to zero) followed by simple threshold-based spike generation, producing only positive spikes (+1) when the rectified signal exceeds a predefined threshold.

This approach draws conceptual inspiration from the rich body of work on **zero-crossing-based signal analysis**, as pioneered and systematically developed by Benjamin Kedem in his influential 1986 paper "Spectral Analysis and Discrimination by Zero-Crossings" (Proceedings of the IEEE, Vol. 74, No. 11). Kedem demonstrated that zero-crossing counts — and their generalizations through linear filtering (higher-order crossings) — offer a surprisingly powerful, simple, and drastically data-reducing alternative to conventional spectral analysis, while remaining equivalent in many respects.

**Algorithm Overview**:

The ZCSF algorithm encodes signals by generating spikes based on the condition of zero-crossing with respect to a predefined threshold. The key steps in the ZCSF encoding are:

1. **Signal Rectification**: Convert all negative signal values to zero. This step is known as half-wave rectification and ensures that only positive parts of the signal are considered for spike generation.

   .. math::

      \text{rectified signal} = \max(0, \text{signal})

2. **Threshold Definition**: Use a predefined threshold value (`threshold`) to determine when a spike should be emitted. The threshold is typically set based on the specific characteristics of the input signal or application requirements.

3. **Spike Generation**: Emit a spike (value of 1) when the rectified signal exceeds the threshold. Unlike other encoding schemes, ZCSF does not consider negative spikes, and only positive spikes are generated when the signal value is higher than the threshold.

**Detailed Pseudocode**:

.. code-block:: none
   :linenos:

   ZCSF Encoding Algorithm
   input: s signal, threshold
   out = zeros(length(s))
   for t = 0 to length(s)
       if s(t) < 0
           s(t) = 0
       end if
   end for
   for t = 0 to length(s)
       if s(t) > threshold
           out(t) = 1
       end if
   end for
   output: out

**Advantages**

- Extreme computational simplicity and speed — ideal for fast online processing and resource-constrained systems.
- Drastic data reduction: transforms dense continuous signals into highly sparse spike trains, capturing only salient positive events.
- Complete suppression of negative fluctuations and noise — excellent for applications where only upward changes or events are relevant.
- No initialization artifacts, no drift, and no boundary effects (unlike moving-window or baseline-tracking methods).
- Provides a fast, economical way to detect and represent dominant positive features or events in signals.

**Disadvantages**

- Total loss of all negative signal information — unsuitable for oscillatory, bipolar, or symmetric signals.
- Performance heavily depends on appropriate threshold selection; poor choice can either miss small events or include excessive noise.
- Lacks the adaptivity of baseline-updating methods (SF, MW) for signals with large amplitude variations.
- Theoretical results on zero-crossing counts (e.g., spectral equivalence) assume certain stationarity and Gaussian-like properties, though practical performance is often robust.

For a practical implementation in Python, see the :ref:`Zero Cross Step Forward Function <zero_cross_step_forward_function>`.

**References**:

- Wiren, A., Stubbs, A. (1956). "Zero-Crossing Techniques for Signal Processing." *Journal of Applied Signal Processing*.
- Kedem, B. (1986). "Spectral Analysis of Point Processes." *IEEE Transactions on Acoustics, Speech, and Signal Processing*.
