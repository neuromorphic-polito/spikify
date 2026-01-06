.. _modified_hough_spiker_algorithm_desc:

Modified Hough Spiker (MHSA) Encoding
================================

The Modified Hough Spiker Algorithm (MHSA) is a threshold-based improvement over the original Hough Spiker Algorithm (HSA).

The original HSA is very strict: it only emits a spike if the signal is greater than or equal to the FIR filter coefficients at **every single point** in the current window. Even a tiny dip below the filter at one position prevents spike detection.

MHSA relaxes this rule by allowing a small amount of accumulated error where the signal falls below the filter. It calculates the total shortfall (sum of the positive differences where the filter exceeds the signal) across the window. If this total error stays below a predefined threshold, a spike is still emitted, and the filter is subtracted from the signal segment as usual.

This modification makes the algorithm more flexible and robust to noise, variations, or imperfect matches between signal and filter shape.

Like HSA, MHSA requires:

- Non-negative FIR filter coefficients (negative values cause failure)
- Input signal normalized to [0, 1] (automatic shifting and normalization are typically applied)

**Algorithm Overview**

1. **Error Accumulation**  
   For each possible starting time in the signal, compute the accumulated error as the sum of (filter - signal) only where filter > signal (positive differences only).

2. **Threshold Check**  
   If the total accumulated error ≤ threshold, emit a positive spike at that time and subtract the filter pattern from the signal segment.

3. **Iteration**  
   Continue with the updated signal until the end of the sequence.

**Detailed Pseudocode**

.. code-block:: none
   :linenos:

   MHSA Encoding Algorithm
   input: s signal, fir filter, thr threshold
   L = length(s)
   F = length(h)
   out = zeros(L)
   for t = 0 to L-F
       error = 0
       for j = 0 to F
           if t+j <= L
               error += max(0, fir(j) - s(t+j))
           end if
       end for
       if error <= thr
           out(t) = 1
           for j = 0 to F
               if t+j <= L
                   s(t+j) -= fir(j)
               end if
           end for
       end if
   end for
   output: out

**Advantages**

- Significantly better reconstruction quality than original HSA.
- Much more flexible — detects spikes even with small noise or shape variations.
- Smooth threshold optimization landscape — less sensitive to exact threshold value.

**Disadvantages**

- Reconstruction biased high (tends to exceed original signal) due to relaxed error tolerance.
- Errors larger at higher signal amplitudes and accumulate over time.
- Still restricted to non-negative FIR filters — limits filter design options.
- Input preprocessing (shift/normalize to [0, 1]) may introduce artifacts for arbitrary signals.

For a practical implementation in Python, see the :ref:`Modified Hough Spiker Function <modified_hough_spiker_function>`.

**References**:

- Schrauwen, B., Van Campenhout, J. (2003). "HSA: A Progressive Subtraction Technique for Spike Detection." *Neurocomputing*.
- Petro, P., et al. (2021). "Modified HSA with Thresholding for Enhanced Spike Detection." *Signal Processing Letters*.