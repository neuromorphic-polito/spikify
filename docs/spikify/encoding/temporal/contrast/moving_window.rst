.. _moving_window_algorithm_desc:

Moving Window (MW) Encoding
================================

The Moving Window (MW) encoding uses a moving baseline with a set threshold value, where the baseline always equals the mean of the preceding signal values in a time window. Thus, the moving baseline is essentially the application of a moving average filter. If the signal value is above or below baseline ± threshold value, a positive or negative spike is registered. MW thus has two parameters: the threshold and the window size. Decoding is essentially the same as for TBR or SF.

**Algorithm Overview**

The algorithm computes a dynamic baseline as the mean of the most recent values within a fixed-size sliding window (or partial window at the beginning). The current signal value is then compared to this baseline ± a fixed threshold to determine whether to emit a spike:

- Positive spike (+1) if s(t) > base + threshold
- Negative spike (-1) if s(t) < base - threshold

The moving average acts as a low-pass filter, providing inherent smoothing and noise reduction.

.. code-block:: none
   :linenos:

   MW Encoding Algorithm
   input: s signal, threshold, window, startpoint
   startpoint = s(0)
   out = zeros(length(s))
   base = zeros(length(s))
   for t = 0:(window)
       if s(t) > base + threshold
           out(t) = 1
       elseif s(t) < base - threshold
           out(t) = -1
       end if
   end for
   for t = (window):length(s)
       base = mean(s(t-window:t))
       if s(t) > base + threshold
           out(t) = 1
       elseif s(t) < base - threshold
           out(t) = -1
       end if
   end for
   output: out

**Advantages**

- Robust against white noise: the moving average baseline acts as an optimal time-domain smoothing filter for white noise.
- Can help attenuate certain artifact frequencies (e.g., power line noise) through appropriate window size selection.
- Adapts locally to signal level changes, suitable for non-stationary signals.

**Disadvantages**

- Trade-off between noise reduction (larger window) and preservation of high-frequency content (smaller window).
- Not recommended for strong narrowband interference (e.g., 50/60 Hz line noise); better to apply a dedicated band-stop filter before encoding.
- Requires tuning of two parameters (threshold and window size), increasing complexity compared to single-parameter methods.
- May delay response to rapid changes due to averaging over past values.

For a practical implementation in Python, refer to the :ref:`Moving Window Function <moving_window_function>`.

**References**:

- Kasabov, N., et al. (2016). "Neural Coding Strategies in Spiking Neural Networks." *Neural Processing Letters*.
- B. Petro, N. Kasabov and R. M. Kiss, "Selection and Optimization of Temporal Spike Encoding Methods for Spiking Neural Networks," in IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 2, pp. 358-370, Feb. 2020, doi: 10.1109/TNNLS.2019.2906158.
