.. _modified_hough_spiker_algorithm_desc:

Modified Hough Spiker Encoding
============================

The Modified Hough Spiker Algorithm (Modified HSA) builds upon the original Hough Spiker Algorithm by introducing a threshold mechanism to handle the detection process. While it maintains the core idea of subtractive, deconvolution-based spike detection, the modified version incorporates a threshold value to accumulate and evaluate errors.

**Algorithm Overview**:

Modified HSA introduces a conditional check on the error accumulation during the spike detection process. This adjustment ensures that only significant deviations from the convolution function are considered as spikes, improving detection accuracy.

1. **Error Accumulation**:

   During the processing of each signal segment, the algorithm accumulates the error between the signal value and the convolution result:

   .. math::

      \text{error} = \text{error} + (\text{filter}[j] - \text{Signal}[i + j - 1])

   Here, `filter[j]` is the value from the convolution result at index `j`, and `Signal[i + j - 1]` is the corresponding signal value.

2. **Threshold Comparison**:

   The key modification in this algorithm is the conditional check on the accumulated error. The subtraction operation from the original HSA is only performed if the accumulated error stays within a specified threshold:

   .. math::

      \text{error} \leq \text{Threshold}

   If this condition is met, the signal is updated by subtracting the convolution value, and a spike is detected.

**Implementation Steps**:

1. **Compute Error**: Calculate the error by comparing the convolution filter values to the corresponding signal values.
2. **Threshold Check**: Compare the accumulated error to the predefined threshold to determine if a spike should be detected.
3. **Progressive Subtraction**: If the threshold condition is met, subtract the filter values from the signal, similar to the original HSA, but with enhanced control over detection.

**Advantages**:

The Modified HSA allows for more flexible spike detection by considering cumulative errors. This approach is beneficial in scenarios where spikes are not distinctly above the convolution function but still significant enough to be detected.

For a practical implementation in Python, see the :ref:`Modified Hough Spiker Function <modified_hough_spiker_function>`.

**References**:

- Schrauwen, B., Van Campenhout, J. (2003). "HSA: A Progressive Subtraction Technique for Spike Detection." *Neurocomputing*.
- Petro, P., et al. (2021). "Modified HSA with Thresholding for Enhanced Spike Detection." *Signal Processing Letters*.