.. _bens_spiker_algorithm_desc:

Ben's Spiker Algorithm (BSA) Encoding
=========================================

An analog signal can be constructed from a spike train by convolution with an FIR filter. BSA (Ben's Spiker Algorithm) is an algorithm for producing the spike train from which the original signal can be reconstructed well. BSA works only for positive-valued signals. First, an FIR filter is created. Then, two error terms are calculated at each time point: one that results from subtracting the filter coefficients from the subsequent signal values, and one that results from not changing the signal. If the subtraction error is smaller than the unchanged signal error term subtracted by a factor, a positive spike is generated and the filter coefficients are subtracted from the signal. Decoding is straightforward since it was kept in mind during the encoding: a convolution of the spike train with the filter coefficients gives the reconstructed signal.

BSA encoding results in a unipolar (only positive) spike train. The original BSA encoding requires input with [0, 1] limits. However, BSA can be applied to any positive-valued signal if the filter coefficients are scaled up such that they appropriately match the signal boundaries. Therefore, a simple signal shift above zero is sufficient.

**Algorithm Overview**

BSA iteratively detects spikes by comparing two cumulative error metrics over a sliding segment of length equal to the FIR filter:

- **error1**: sum of absolute differences between the current signal segment and the filter coefficients
- **error2**: sum of absolute values of the current signal segment

A spike is generated if:

.. math::

   \text{error1} \leq \text{error2} - \text{threshold}

When a spike is detected, the filter is subtracted from the corresponding segment of the signal, effectively removing the detected spike pattern for subsequent iterations.

**Detailed Pseudocode**

.. code-block:: none
   :linenos:

   BSA Encoding Algorithm
   input: s signal, fir filter, thr threshold
   L = length(s)
   F = length(fir)
   out = zeros(L)
   shift = min(s)
   s = s - shift
   for t = 0:(L-F+1)
       err1 = 0
       err2 = 0
       for k = 0:F
           err1 = err1 + abs(s(t+k) - fir(k))
           err2 = err2 + abs(s(t+k))
       end for
       if err1 <= (err2 - thr)
           out(t) = 1
           for k = 1:F
               s(t+k) = s(t+k) - fir(k)
           end for
       end if
   end for
   output: out, shift

**Advantages**

- Designed with reconstruction in mind: the original signal can be well recovered via simple FIR convolution.
- Robust representation of continuously changing, smooth, and trended signals.
- Simultaneously performs filtering during encoding.
- Allows flexible scaling of filter coefficients to improve dynamic range and reduce saturation.
- Good SNR for signals within the designed filter bandwidth.

**Disadvantages**

- Only suitable for positive-valued signals (requires shifting/preprocessing for general inputs).
- Poor performance on constant plateaus and step-like changes: requires constant firing to maintain nonzero values, which can fail or saturate, especially at higher levels.
- Significant errors at signal start (catch-up from 0) and end (convolution tail equal to filter length) — can cause substantial information loss in short signals.
- Introduces offset and scaling errors in reconstruction depending on optimization.
- Can generate low-frequency artifact components, especially with large filter sizes.
- Non-smooth optimization landscape with multiple local peaks — parameter search (cutoff, filter size, threshold) is nontrivial.
- Higher plateaus are particularly problematic if filter coefficients are not scaled up appropriately (e.g., sum of coefficients ≈ 2 × signal max recommended).

**References**

- Schrauwen, B., Van Campenhout, J. (2003). "BSA: An Efficient Algorithm for Time-Critical Signal Processing." *Neurocomputing*.
- Petro, P., et al. (2020). "Revisiting the BSA for Modern Applications." *Signal Processing Letters*.
- B. Petro, N. Kasabov and R. M. Kiss, "Selection and Optimization of Temporal Spike Encoding Methods for Spiking Neural Networks," in IEEE Transactions on Neural Networks and Learning Systems.