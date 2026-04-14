.. _threshold_based_representation_algorithm_desc:

Threshold-Based Representation (TBR) Encoding
=============================================

Threshold-Based Representation (TBR) is one of the simplest and most foundational methods in temporal contrast encoding for spiking neural networks. TBR generates spikes by tracking significant temporal changes in the signal. A positive or negative spike is emitted when the variation between consecutive signal values exceeds a dynamically computed threshold, with polarity determined by the sign of the change.

The threshold is adaptive to the signal's characteristics: it is computed over the entire sequence by taking the first-order differences, then setting the threshold as the mean of these variations plus a tunable factor multiplied by their standard deviation. This makes the encoding largely independent of absolute signal amplitude while effectively suppressing noise. The single hyperparameter factor controls sensitivity lower values preserve more variations, while higher values emit spikes only for substantial events.

Decoding is straightforward: the original signal can be reconstructed by cumulatively summing the spikes (each scaled by the encoding threshold with appropriate sign), starting from the initial signal value.

**Algorithm Overview**

The TBR encoding method processes a signal (possibly with multiple channels), evaluating variations across each channel between consecutive time steps. The main steps are:

1. **Compute Variations**: Calculate the difference between consecutive time steps for each channel.
2. **Define Threshold**: For each channel, compute the threshold using the formula:

   .. math::

      \text{threshold} = \text{mean}(\text{diff}) + \text{factor} \cdot \text{std}(\text{diff})

   where factor controls the noise-reduction band:

   - factor = 0: All signal variations are kept.
   - 0 < factor ≤ 1: Small variations are filtered out, preserving major signal changes.
   - factor > 1: Significant noise reduction; only major variations generate spikes.

3. **Determine Spikes**: Emit +1 if diff > threshold, -1 if diff < -threshold, else 0.
4. **Construct the Spike Train**: Build the output spike train with values +1, -1, or 0.

**Detailed Pseudocode**

.. code-block:: none
   :linenos:

   TBR Encoding Algorithm
   input: s signal, f factor
   startpoint = s(0)
   diff = zeros(length(s))
   for t = 0:length(s)-1
       diff(t) = s(t+1) - s(t)
   end for
   diff(end) = diff(end-1)
   threshold = mean(diff) + f*std(diff)
   out = zeros(length(s))
   for t = 0:length(s)
       if diff(t) > threshold
           out(t) = 1
       elseif diff(t) < -threshold
           out(t) = -1
       end if
   end for
   output: out, threshold

**Advantages**

- Simple and computationally efficient, originally designed for fast online/streaming processing.
- Effectively reduces small perturbations and white noise by thresholding minor variations.
- No artifacts (false spikes) at the start or end of the spike train or reconstructed signal.
- Amplitude-independent encoding; threshold adapts to signal characteristics.
- Straightforward and exact decoding via cumulative summation starting from the initial value.

**Disadvantages**

- Small, gradual changes are ignored; only large enough variations generate spikes.
- Poor representation of sudden step-wise changes due to uniform reconstruction step size equal to the threshold.
- Introduces scaling errors, especially prominent in trended signals.
- Trade-off in factor selection: low factor captures small events but includes noise; high factor misses small events but filters noise.
- Global threshold (computed over the entire sample) can be suboptimal for long signals with varying amplitude dynamics across different segments.
- Sensitive to strong white noise, which can mask gradual changes and introduce strong low-frequency (1/frq or "pink") artifacts during reconstruction, leading to drift in longer signals.
- Parameter optimization is challenging—multiple local minima and plateaus in error landscapes due to events of differing amplitudes, making automatic tuning unreliable without domain knowledge.

For a practical implementation in Python, see the :ref:`Threshold Based Representation Function <threshold_based_representation_function>`.

**References**

- Delbruck, T., Lichtsteiner, P. (2007). "Artificial Retina: Applications of Image Processing with Spiking Neural Networks." *IEEE Transactions on Neural Networks*.
- B. Petro, N. Kasabov and R. M. Kiss, "Selection and Optimization of Temporal Spike Encoding Methods for Spiking Neural Networks," in IEEE Transactions on Neural Networks and Learning Systems.