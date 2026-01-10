.. _poisson_algorithm_desc:

Poisson Encoding
================

Poisson encoding is the classical and most widely used **rate coding** strategy for converting continuous signals or static values into discrete, irregular spike trains. It models spike generation as an **inhomogeneous Poisson process**, where the instantaneous firing rate :math:`r(t)` is directly proportional to the input intensity at time :math:`t`.

This method is biologically inspired by the irregular, rate-modulated firing observed in many sensory neurons (e.g., retinal ganglion cells, auditory nerve fibers) and is a standard choice for benchmarking spiking neural networks (SNNs), especially when converting static datasets like made of images to spike trains.

**Algorithm Overview**

Spikes are generated independently with probability proportional to the input value and time bin size. The probability of observing exactly :math:`k` spikes in a small interval :math:`\Delta t` follows the Poisson distribution:

.. math::

   P(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad \lambda = r(t) \cdot \Delta t

where:

- :math:`r(t)` is the instantaneous firing rate, usually scaled from the normalized input signal,
- :math:`k` is the expected number of spikes in the time window,
- :math:`\Delta t` is the discrete time step.

**Detailed Pseudocode**

.. code-block:: none
   :linenos:

   Algorithm Poisson Encoding
   input: s signal, Δt unterval length
   out = zeros(length(s))
   n_blocks = T // Δt
   blocks = reshape(s, (n_blocks, Δt))
   rates = mean(blocks, axis=1)
   for b = 0:Δt
        if rate[b] > 0:
            U = uniform(0,1, size=Δt)
            ISI = -log(1 - U) / (rate[b] * Δt)
            cumsum = cumulative_sum(ISI)
            steps = searchsorted(bins, cumsum) - 1
            for s = 0:length(steps)
                if steps[s] < Δt
                    t = b * Δt + step[s]
                    out[t] = 1
                end if
            enf for
        end if
    end for
   output: out

**Advantages**

- Biologically plausible — reproduces the irregular, rate-modulated firing observed in many cortical and sensory neurons.
- Simple, analytically tractable, and easy to implement.
- Standard for benchmarking SNNs (e.g., converting MNIST to spike trains for rate-based training).

**Disadvantages**

- Extremely dense spike trains — high energy consumption and synaptic events in neuromorphic hardware.
- Complete loss of precise temporal information — only average rate matters.
- High latency — many spikes required for accurate rate estimation.
- Not event-driven — generates spikes even during constant input, unlike temporal contrast methods (TBR, SF, etc.).
- Less efficient than sparse, deterministic encodings (e.g., TTFS, rank-order) for rapid processing or low-power applications.

For details on how to implement this algorithm in Python, refer to the :ref:`Poisson Function <poisson_function>`.

**References**

- Auge, D., Hille, J., Mueller, E., and Knoll, A. (2021). A survey of encoding techniques for signal processing in spiking neural networks. Neural Process.
- Liu, Q., Pineda-García, G., Stromatias, E., Serrano-Gotarredona, T., and Furber, S. B. (2016). Benchmarking spike-based visual recognition: a dataset and evaluation. Front. Neurosci.
