.. _poisson_rate_algorithm_desc:

Poisson Rate Encoding
======================

Poisson rate encoding is a neural encoding strategy used to convert continuous signals into discrete spike trains using a Poisson distribution. This method is effective in neural modeling, particularly in rate coding strategies where the spike rate corresponds to the intensity of the input signal.

**Algorithm Overview**:

The Poisson rate encoding uses a Poisson distribution to model the probability of emitting a certain number of spikes (:math:`n`) over a given time interval (:math:`\Delta t`). The probability of having :math:`n` spikes in an interval :math:`\Delta t` is described by the formula:

.. math::

   P(n | \Delta t) = \frac{(r \Delta t)^n}{n!} e^{-r \Delta t} \quad (1)

where:

- :math:`r` is the spike rate, representing the real value to be encoded,
- :math:`n` is the number of spikes,
- :math:`\Delta t` is the time interval.

**Implementation Steps**:

The implementation of this algorithm can be performed through the following steps:

1. **Define the Time Interval** (:math:`\Delta t`): Determine the interval within which to generate the spike train.
2. **Generate a Sequence of Random Numbers** (:math:`x \in [0, 1] \subset \mathbb{R}`): Create a series of random numbers uniformly distributed in the range `[0, 1]`.
3. **Compute Spike Times** (:math:`t_i`): Starting from :math:`t = 0`, calculate the spike times :math:`t_i` as:

   .. math::

      t_i = t_{i-1} + ISI_i \quad \text{for } i \geq 1 \quad (2)

   where the inter-spike interval (:math:`ISI_i`) is defined as:

   .. math::

      ISI_i = -\frac{\log(1 - x_i)}{r} \quad (3)

   This represents the :math:`i`-th inter-spike interval, the time interval in which the probability of having `n = 0` spikes equals :math:`x_i`.
4. **Generate Spikes**: A spike is emitted at each calculated spike time :math:`t_i` until :math:`t_i > \Delta t`.

**Applications**:

Poisson rate encoding is utilized in various neural network models that require a spike-based representation of continuous input data. This encoding method is particularly useful in simulating sensory neurons and other neural circuits where input signals need to be transformed into spike trains suitable for spiking neural networks (SNNs).

For details on how to implement this algorithm in Python, refer to the :ref:`Poisson Rate Function <poisson_rate_function>`.

**References**:

- Auge, J., et al. (2021). "Rate Coding Strategies in Neural Networks." *Neural Computation*.
- Liu, Y., et al. (2016). "Poisson Spike Train Generation for Neural Computation." *Neural Networks*.
