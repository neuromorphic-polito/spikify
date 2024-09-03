.. _zero_cross_step_forward_algorithm_desc:

Zero-Crossing Step-Forward (ZCSF) Encoding
==========================================

The Zero-Crossing Step-Forward (ZCSF) algorithm is a variant of the Step-Forward (SF) encoding technique that utilizes zero-crossings of the signal. Unlike the standard SF method, which involves a baseline (Base) value, ZCSF relies on a half-wave rectification behavior and emits spikes only for positive signal values exceeding a specified threshold.

**Algorithm Overview**:

The ZCSF algorithm encodes signals by generating spikes based on the condition of zero-crossing with respect to a predefined threshold. The key steps in the ZCSF encoding are:

1. **Signal Rectification**: Convert all negative signal values to zero. This step is known as half-wave rectification and ensures that only positive parts of the signal are considered for spike generation.

   .. math::

      \text{Rectified Signal} = \max(0, \text{Signal})

2. **Threshold Definition**: Use a predefined threshold value (`Threshold`) to determine when a spike should be emitted. The threshold is typically set based on the specific characteristics of the input signal or application requirements.

3. **Spike Generation**: Emit a spike (value of 1) when the rectified signal exceeds the threshold. Unlike other encoding schemes, ZCSF does not consider negative spikes, and only positive spikes are generated when the signal value is higher than the threshold.

**Implementation Steps**:

To implement the Zero-Crossing Step-Forward encoding in Python:

1. **Rectify the Signal**: Use `numpy`'s `maximum` function to zero out all negative signal values.
2. **Apply Threshold Condition**: Generate spikes by checking if the rectified signal exceeds the given threshold.
3. **Construct the Spike Train**: Create an output array that holds the spike values, where spikes occur based on the defined conditions.

**Advantages**:

The ZCSF encoding method is particularly useful for applications where only significant positive changes in the signal are of interest. By ignoring negative values and focusing on zero-crossings with respect to a threshold, ZCSF can efficiently represent signals in scenarios like event-based processing or certain types of sensory data interpretation.

For a practical implementation in Python, see the :ref:`Zero Cross Step Forward Function <zero_cross_step_forward_function>`.

**References**:

- Wiren, A., Stubbs, A. (1956). "Zero-Crossing Techniques for Signal Processing." *Journal of Applied Signal Processing*.
- Kedem, B. (1986). "Spectral Analysis of Point Processes." *IEEE Transactions on Acoustics, Speech, and Signal Processing*.
