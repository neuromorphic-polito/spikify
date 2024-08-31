.. _poisson_rate_algorithm_desc:

Poisson Rate Encoding
======================

Poisson rate encoding is a neural encoding strategy used to convert continuous signals into discrete spike trains. This method is widely used in computational neuroscience and neural networks to simulate the firing behavior of neurons.

**Algorithm Overview**:

Poisson rate encoding uses a Poisson distribution to model the probability of emitting spikes over time. The rate parameter (`r`) of the Poisson distribution determines how frequently spikes occur within a specified time interval (`Δt`).

The probability of observing `n` spikes in a time interval `Δt` is given by the Poisson distribution formula:

.. math::

    P(n | \Delta t) = \frac{(r \Delta t)^n}{n!} e^{-r \Delta t}

where:

- `r` is the rate of the signal (the real value to be encoded),

- `n` is the number of spikes,

- `Δt` is the time interval.

**Steps of the Algorithm**:

1. **Define the Time Interval (`Δt`)**: Choose the time interval over which to generate the spike train.
2. **Generate Random Numbers (`x ∈ [0, 1]`)**: Create a sequence of random numbers uniformly distributed between 0 and 1.
3. **Compute Inter-Spike Intervals (ISIs)**: For each time step `t`, compute the next spike time `t_i` using the inter-spike interval `ISI_i`:

   .. math::

       ISI_i = -\frac{\log(1 - x_i)}{r}

   The spike times `t_i` are calculated by summing the inter-spike intervals.
4. **Generate Spikes**: Emit a spike at each computed spike time :math:`t_i` until `t_i > Δt`.

**Applications**:

Poisson rate encoding is used in various neural models where spike-based representations are required. It is especially useful in simulating sensory neurons and encoding continuous inputs into a format suitable for spiking neural networks (SNNs).

For more details on implementing this algorithm, see the :ref:`Poisson Rate Algorithm <poisson_rate>` module.

**References**:

- Auge, J., et al. "Rate Coding Strategies in Neural Networks." Neural Computation, 2021.
- Liu, Y., et al. "Poisson Spike Train Generation for Neural Computation." Neural Networks, 2016.
