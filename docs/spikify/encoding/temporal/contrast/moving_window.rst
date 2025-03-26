.. _moving_window_algorithm_desc:

Moving Window Encoding
=======================

The Moving Window (MW) encoding algorithm is another strategy used in neural modeling to convert continuous signals into discrete spike trains. Unlike the Poisson rate encoding, which relies on a probabilistic approach, the Moving Window method uses a sliding window mechanism and thresholding to detect spikes.

**Algorithm Overview**:

The Moving Window algorithm utilizes a sliding window approach to compute the average value (referred to as the "Base") of the signal within a window of fixed length. It then uses a threshold value to determine when spikes occur based on deviations from this baseline. The formulas used in this algorithm are:

.. math::

   \text{Threshold} = \text{mean}(\text{Variation}) \quad (5)

.. math::

   \text{Base} = \text{mean}(\text{Signal}[1:\text{Window}]) \quad (6)

where:

- **Threshold**: The mean variation within the signal.
- **Base**: The mean of the signal values within the current sliding window.

**Implementation Steps**:

To implement this algorithm, follow these steps:

1. **Set the Sliding Window Length** (:math:`\text{Window Length}`): Define the window length within which the baseline (`Base`) is computed.
2. **Calculate the Base**: For each position in the signal, compute the mean value (`Base`) of the signal values within the current window. If the window extends beyond the start of the signal, use the available values up to the current position.
3. **Apply the Threshold**: For each signal value at time :math:`t`, determine whether a spike occurs by comparing the value to `Base + Threshold` and `Base - Threshold`.
   - If the signal exceeds `Base + Threshold`, a positive spike (+1) is generated.
   - If the signal falls below `Base - Threshold`, a negative spike (-1) is generated.
4. **Generate the Spike Train**: Construct the spike train by marking the time points where spikes occur.

**Advantages**:

The Moving Window algorithm is more robust to noise compared to the TBR (Time-Based Rate) method because it directly relies on the signal's absolute value rather than its variation. This approach helps to reduce false positives and negatives in noisy conditions (Kasabov et al., 2016).

For a practical implementation in Python, refer to the :ref:`Moving Window Function <moving_window_function>`.

**References**:

- Kasabov, N., et al. (2016). "Neural Coding Strategies in Spiking Neural Networks." *Neural Processing Letters*.
