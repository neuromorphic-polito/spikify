.. _hough_spiker_algorithm_desc:

Hough Spiker (HSA) Encoding
============================

The Hough Spiker Algorithm (HSA) is a parameter-free technique for converting non-negative analog signals (normalized to [0, 1]) into unipolar spike trains that can be reconstructed via convolution with the same FIR filter used during encoding. At each time step, the algorithm compares the shifted FIR filter impulse response pointwise with the matching segment of the input signal. If fir filter is everywhere smaller or equal to segment signal, a positive spike is emitted, and fir filter is subtracted from the signal to remove the detected pattern. This process is repeated sequentially across the signal.

HSA assumes a FIR filter with non-negative coefficients (e.g., low-pass filters without negative taps), as negative values would cause the algorithm to fail. The input range is limited to [0, 1], so signals are typically shifted and normalized if necessary.

**Algorithm Overview**

The core idea is to reverse the decoding convolution: iteratively detect and subtract filter patterns that "fit" under the signal curve.

1. **Pointwise Comparison**:
   For each possible starting time, check whether the signal values in the window are greater than or equal to the corresponding filter coefficients at every single position.

2. **Spike Emission and Subtraction**:
   If the condition holds (i.e., signal ≥ filter everywhere in the window), emit a positive spike (+1) at time t and subtract the filter coefficients from the signal segment.

This results in a sparse unipolar spike train, but reconstruction is biased downward due to the strict "≥ filter" condition.

**Detailed Pseudocode**

.. code-block:: none
   :linenos:

    Algorithm HSA Encoding
    input: s signal, fir filter
    L = length(s)
    F = length(h)
    out = zeros(L)
    for t = 0 to L-F+1
        count = 0
        for j = 0 to F
            if t+j < L
                if s(t+j) >= fir(j)
                    count += 1
                end if
            end if
        end for
        if count == F
            out(t) = 1
            for j = 0 to F
                if t+j < L
                    s(t+j) -= fir(j)
                end if
            end for
        end if
    end for
    output: out

**Advantages**

- Extremely simple and parameter-free — no tuning required beyond filter design.
- Computationally lightweight and suitable for real-time encoding.
- Produces unipolar spike trains reconstructible via simple FIR convolution.
- Effective for signals within the filter's bandwidth, with minimal external parameters.
- No threshold optimization needed, unlike modified HSA or BSA.

**Disadvantages**

- Limited to non-negative FIR filters — cannot use steep filters with negative taps, restricting bandwidth and sharpness.
- Reconstruction is biased: converted signal always stays below the original due to strict "≥ filter" condition.
- Restricted to non-negative FIR filters — limits filter design options.
- Errors increase with higher signal values and persist over time (cumulative bias).
- Requires non-negative, [0, 1]-normalized inputs — preprocessing (shift/normalize) may be needed, potentially introducing artifacts.
- Less flexible than threshold-based variants (modified HSA) or error-minimizing methods (BSA) for diverse signals.

For a practical implementation in Python, see the :ref:`Hough Spiker Function <hough_spiker_function>`.

**References**

- Schrauwen, B., Van Campenhout, J. (2003). "HSA: A Progressive Subtraction Technique for Spike Detection." *Neurocomputing*.
- Petro, P., et al. (2020). "Revisiting the HSA for Modern Applications." *Signal Processing Letters*.
