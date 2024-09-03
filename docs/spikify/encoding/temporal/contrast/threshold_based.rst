.. _threshold_based_representation_algorithm_desc:

Threshold-Based Representation (TBR) Encoding
=============================================

The Threshold-Based Representation (TBR) algorithm is a method for encoding signals by generating spikes based on absolute signal variations relative to a fixed threshold. This technique is particularly useful for reducing noise and emphasizing significant changes in the signal.

**Algorithm Overview**:

The TBR encoding method processes a signal composed of multiple channels, evaluating variations across each channel between consecutive time steps. A specific threshold is defined to determine when a spike should be generated. The main steps are:

1. **Compute Variations**: For a signal with multiple channels, calculate the variation (`Variation`) along each channel between consecutive time steps.
2. **Define Threshold**: For each channel, compute a threshold using the formula:

   .. math::

      \text{Threshold} = \text{mean}(\text{Variation}) + \gamma \cdot \text{std}(\text{Variation}) \quad (4)

   where:

   - **Variation**: The difference in signal value between consecutive time steps.
   - **Threshold**: A dynamic value based on the mean and standard deviation of the signal variations, adjusted by a tunable parameter :math:`\gamma`.
   - :math:`\gamma`: A parameter that controls the noise-reduction band. Depending on the noise level to be filtered out, different values for :math:`\gamma` can be selected:
     - :math:`\gamma = 0`: All signal variations are kept.
     - :math:`0 < \gamma \leq 1`: Small variations are filtered out, preserving major signal changes.
     - :math:`\gamma > 1`: Significant noise reduction, allowing only major variations to generate spikes.

3. **Determine Spikes**: For each time step, if the absolute value of `Variation` exceeds the `Threshold`, a spike is generated with polarity determined by the sign of both `Variation` and `Threshold`.

4. **Construct the Spike Train**: Generate a spike train by assigning spike values (+1 or -1) based on the conditions outlined above.

**Implementation Steps**:

To implement the Threshold-Based Representation in Python:

1. Compute the variation of the signal using `numpy`'s `diff` function.
2. Calculate the `Threshold` using the mean and standard deviation of the variations, adjusted by the parameter :math:`\gamma`.
3. Apply conditions to determine where spikes occur based on the computed threshold.
4. Generate the output spike train array.

**Advantages**:

The TBR algorithm is effective for emphasizing significant changes in a signal while filtering out minor variations, making it ideal for applications requiring robust noise reduction.

For a practical implementation in Python, see the :ref:`Threshold Based Representation Function <threshold_based_representation_function>`.

**References**:

- Delbruck, T., Lichtsteiner, P. (2007). "Artificial Retina: Applications of Image Processing with Spiking Neural Networks." *IEEE Transactions on Neural Networks*.
