.. _bens_spiker_algorithm_desc:

Ben's Spiker Encoding
============================

Bens Spiker Algorithm is a method used to detect spikes within a signal by comparing two cumulative error metrics. These metrics are calculated over a segment of the signal, which is filtered using a boxcar (or rectangular) window of a specified length. The algorithm relies on the comparison of these errors to determine whether a spike is present at a given timestep.

**Algorithm Overview**:

Bens Spiker Algorithm works as follows:

1. **Cumulative Error Calculation**:
   
   For each segment of the signal (of length equal to the boxcar window), two cumulative errors are calculated:

   .. math::
      \text{error1} = \sum_{j=1}^{n} \left| \text{Signal}[i+j-1] - \text{filter}[j] \right|


   .. math::
      \text{error2} = \sum_{j=1}^{n} \left| \text{Signal}[i+j-1] \right|


Here, `error1` is the sum of absolute differences between the signal segment and the boxcar filter window, while `error2` is the sum of absolute values of the signal segment itself.

2. **Spike Detection Condition**:

   After calculating `error1` and `error2`, a condition is checked to determine whether a spike should be emitted:

   .. math::

      \text{error1} \leq \text{error2} \cdot \text{Threshold}

   If the above condition holds true, it indicates that the difference between the signal and the filter is small enough that the signal is close to the filter shape, thereby detecting a spike at that position.

3. **Signal Adjustment**:

   Upon detection of a spike, the corresponding segment of the signal is adjusted by subtracting the filter window from it, allowing the algorithm to process subsequent spikes in the signal.

**Implementation Steps**:

1. **Create the Boxcar Filter Window**: A boxcar window of specified length is used to smooth the signal.
2. **Iterate Through the Signal**: Calculate the cumulative errors `error1` and `error2` for each segment of the signal.
3. **Detect and Record Spikes**: Check the spike detection condition and update the signal and spike array accordingly.

**Advantages**:

Ben's Spiker Algorithm is robust for detecting spikes in signals where the spike characteristics closely match the shape of the filter window. This makes it particularly suitable for signals with regular or repetitive spike-like features.

For a practical implementation in Python, see the :ref:`Ben Spiker Function <bens_spiker_function>`.

**References**:

- Schrauwen, B., Van Campenhout, J. (2003). "BSA: An Efficient Algorithm for Time-Critical Signal Processing." *Neurocomputing*.
- Petro, P., et al. (2020). "Revisiting the BSA for Modern Applications." *Signal Processing Letters*.
