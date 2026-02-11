.. _burst_coding_algorithm_desc:

Burst Coding (BC)
===================

Burst Coding is a biologically inspired rate-like coding strategy that represents input intensity using **two characteristics** of a spike burst: the **number of spikes** (:math:`N_s`) and the **inter-spike interval** (ISI). Stronger inputs produce bursts with more spikes and shorter ISIs, while weaker inputs produce fewer spikes with longer ISIs.

This dual-parameter encoding increases reliability and information density, as it leverages both spike count and precise timing within short bursts. It is particularly effective for rapid, robust transmission in sensory pathways and spiking neural networks (SNNs).

**Algorithm Overview**

The input intensity P (normalized to [0, 1]) determines:

- Number of spikes per burst:  

  .. math::
      N_s(P) = \lceil P \cdot N_{\max} \rceil

- Inter-spike interval (ISI):  

  .. math::
     \text{ISI}(P) = 
     \begin{cases}
       t_{\max} & \text{if } N_s = 1 \\
       \left\lceil (t_{\max} - t_{\min}) \cdot (1 - P) + t_{\min} \right\rceil & \text{otherwise}
     \end{cases}

Larger P produces bursts with more spikes and smaller ISIs. The burst is placed at regular intervals within a time window sufficient to accommodate the longest burst.

**Detailed Pseudocode**

.. code-block:: none
   :linenos:

   Burst Coding Algorithm
   input: s signal (normalized P ∈ [0,1]), N_max max spikes, t_min min ISI, t_max max ISI, interval_length output window
   out = zeros(length(s))

   n_blocks = length(s)
   for block_idx = 0 to n_blocks-1
       P = mean(s[block_idx * interval_length : (block_idx+1) * interval_length])

       N_s = ceil(P * N_max)
       ISI = ceil((t_max - t_min) * (1 - P) + t_min)
       

       spike_times = arange(0, N_s * (ISI + 1), ISI + 1)
       spike_times = spike_times[spike_times < interval_length]

       for time in spike_times:
           global_t = block_idx * interval_length + time
           out[global_t] = 1
       end for
   end for

   output: out

**Advantages**

- Higher information density and reliability in which both count and timing are used.
- Biologically plausible — mimics burst firing in sensory neurons for rapid, robust transmission.
- More robust to noise.
- Efficient for classification/recognition tasks.

**Disadvantages**

- Requires sufficient time window per burst.
- Sensitive to parameter choice (N_max, t_min/t_max) and block length.
- Reconstruction more complex than pure rate (needs both spike count and ISI measurement).

For a practical implementation in Python, see the :ref:`Burst Coding Function <burst_coding_function>`.

**References**

- Guo et al. (2021). "Neural coding in spiking neural networks: a comparative study for robust neuromorphic systems" *Front. Neurosci.*
