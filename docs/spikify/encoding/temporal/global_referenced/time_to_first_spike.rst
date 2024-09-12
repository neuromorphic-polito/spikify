.. _time_to_first_spike_algorithm_desc:

Time-to-First-Spike Encoding
============================

The Time-to-First-Spike (TTFS) encoding method focuses on the time it takes for a neuron to fire its first spike in response to a stimulus. This technique was explored by Rueckauer and Liu (2018) and has been further developed in Park et al. (2020). In our implementation, we employ a dynamic threshold that decays exponentially over time, defined by a membrane potential.

**Algorithm Overview**:

The TTFS encoding process operates as follows:

1. **Dynamic Thresholding**:
   The membrane potential is modeled as an exponentially decaying function, described by the formula:
   
   .. math::

       P_{th}(t) = \theta_0 e^{-t/\tau_{th}}

   where :math:`\theta_0` is a constant and :math:`\tau_{th}` is the decay time constant. For our investigation, we set :math:`\theta_0 = 1` and :math:`\tau_{th} = 0.1`.

2. **Spike Timing Calculation**:
   The input signal is normalized, and based on its intensity, the time to the first spike is determined by comparing the input signal to the dynamic threshold. The lower the signal, the earlier the spike occurs.

3. **Quantization**:
   Similar to the phase encoding approach, the spike times are quantized into discrete levels, which map the signal intensity to binary values.

4. **Binary Representation**:
   The spike times are then converted into a binary sequence, representing the encoded signal in a spike train format.

**Implementation Steps**:

1. **Normalize the Signal**: The input signal is rectified and normalized between 0 and 1.
2. **Compute Dynamic Threshold**: Use an exponentially decaying threshold function to model membrane potential decay.
3. **Determine Spike Timing**: Calculate the time to first spike based on the signal intensity and dynamic threshold.
4. **Quantize Spike Times**: Discretize the spike times and assign binary values.
5. **Generate Spike Train**: Convert the spike timing into a binary spike train for output.

**Advantages**:

TTFS encoding provides a biologically plausible way to represent the intensity of input stimuli by the timing of a single spike. The dynamic threshold approach ensures that even small differences in input intensity can result in distinguishable spike times.

For a practical implementation in Python, see the :ref:`Time-to-First-Spike Encoding Function <time_to_first_spike_function>`.

**References**:

- Rueckauer, B., & Liu, S.-C. (2018). "Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification." *Frontiers in Neuroscience*.
- Park, S., et al. (2020). "Dynamic Threshold Spike Encoding for Robust Neural Coding." *Neural Computation*.
