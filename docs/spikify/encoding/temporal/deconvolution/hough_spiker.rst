.. _hough_spiker_algorithm_desc:

Hough Spiker Encoding
============================

The Hough Spiker Algorithm is a technique used to detect spikes in a signal by performing a progressive subtraction operation. The algorithm first compares the value of the analog signal to the result of a convolution operation. If the signal value exceeds the convolution result, the signal undergoes a subtraction of the convolution value. This process is repeated iteratively for each signal channel.

**Algorithm Overview**:

Hough Spiker algorithm follows these steps:

1. **Progressive Subtraction**:

   For each signal segment, the algorithm compares the signal value at each time step with the result of a convolution operation (using a rectangular window in our analysis):

   .. math::

      \text{Signal}[i+j-1] = \text{Signal}[i+j-1] - \text{filter}[j]

   Here, `Signal[i+j-1]` represents the value of the signal at a specific time step, and `filter[j]` is the value from the convolution result at the corresponding index `j`.

2. **Spike Detection**:

   A spike is detected if the signal value surpasses the convolution result. The signal is then updated by subtracting the filter window from it. This step is performed iteratively across the entire signal, ensuring that spikes are detected and adjusted accordingly.

**Implementation Steps**:

1. **Create the Boxcar Filter Window**: A boxcar window of the specified length is used for the convolution operation.
2. **Iterate Through the Signal**: Compare the signal values with the filter window and update the signal if a spike is detected.
3. **Record Detected Spikes**: A spike array is maintained, where a `1` indicates a detected spike, and `0` indicates no spike.

**Advantages**:

The Hough Spiker Algorithm is efficient in detecting spikes in signals, particularly when the spikes closely match the convolution filter shape. It is especially useful for signals where spikes need to be detected and progressively subtracted to prevent overlap in detection.

For a practical implementation in Python, see the :ref:`Hough Spiker Function <hough_spiker_function>`.

**References**:

- Schrauwen, B., Van Campenhout, J. (2003). "HSA: A Progressive Subtraction Technique for Spike Detection." *Neurocomputing*.
- Petro, P., et al. (2020). "Revisiting the HSA for Modern Applications." *Signal Processing Letters*.
