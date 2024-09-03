.. _step_forward_algorithm_desc:

Step Forward Encoding
=====================

The Step-Forward (SF) encoding algorithm is a refinement of the Moving Window approach and is designed to improve the robustness and accuracy of spike generation in response to signal changes. Proposed by Kasabov et al. (2016), this method iteratively updates the baseline value (`Base`) and uses a threshold to determine spike events.

**Algorithm Overview**:

The Step-Forward algorithm computes an updated baseline (`Base`) for each signal value, which adjusts dynamically based on signal changes. A `Threshold` value determines when a spike should be generated. The formulas used in this algorithm are:

.. math::

   \text{Threshold} = \frac{\text{mean}(\text{Jump})}{\gamma} \quad (7)

.. math::

   \text{Base} = \text{Signal}[1] \quad (8)

where:

- **Threshold**: A dynamic value calculated from the mean of the "jump" (maximum-to-minimum differences in the signal) divided by a tunable parameter :math:`\gamma`.
- **Base**: The initial value of the signal used to track changes dynamically.

**Implementation Steps**:

To implement this algorithm, follow these steps:

1. **Initialize the Base Value** (:math:`\text{Base}`): Set the `Base` to the first value of the input signal.
2. **Iterate Over Signal**: For each signal value at time :math:`t`, compare the current signal value to the dynamically updated `Base` plus or minus the `Threshold`.
   - If the signal exceeds `Base + Threshold`, generate a positive spike (+1) and update `Base` to `Base + Threshold`.
   - If the signal falls below `Base - Threshold`, generate a negative spike (-1) and update `Base` to `Base - Threshold`.
3. **Generate the Spike Train**: Construct the spike train by recording the time points where spikes occur based on the step-forward logic.

**Advantages**:

The Step-Forward algorithm is highly adaptable to changes in signal magnitude and direction, making it particularly effective in environments with fluctuating data. It offers better noise resistance and finer control over spike generation compared to simpler threshold-based methods (Kasabov et al., 2016).

For a practical implementation in Python, refer to the :ref:`Step Forward Function <step_forward_function>`.

**References**:

- Kasabov, N., et al. (2016). "Neural Coding Strategies in Spiking Neural Networks." *Neural Processing Letters*.
- Delbruck, T., Lichtsteiner, P. (2007). "Artificial Retina: Applications of Image Processing with Spiking Neural Networks." *IEEE Transactions on Neural Networks*.
