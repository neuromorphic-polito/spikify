.. _time_to_first_spike_algorithm_desc:

Time-to-First-Spike (TTFS)
======================================

Time-to-First-Spike (TTFS) encoding is a sparse temporal coding strategy where input intensity is represented by the **timing of a single first spike** within a fixed time window. Stronger inputs trigger earlier spikes, enabling ultra-rapid information transmission with minimal spikes (typically one per neuron or input block).

This method is biologically plausible for fast sensory responses (e.g., tactile stimuli in milliseconds) and is highly efficient for neuromorphic hardware, as it requires very few synaptic events. It approximates integration against an exponentially decaying threshold, with spike latency inversely related to input strength.

**Algorithm Overview**

TTFS uses a dynamic, exponentially decaying threshold to model membrane potential:

.. math::

   P_{th}(t) = \theta_0 e^{-t / \tau_{th}}

A spike is emitted at the earliest time t where the normalized input exceeds the threshold. In practice, intensity is mapped to latency via an inverse (logarithmic) function, and the signal is processed in blocks with one spike per block.

**Detailed Pseudocode**

.. code-block:: none
   :linenos:

   Time-to-First-Spike Encoding Algorithm
   input: s signal (length T), interval_length (block size)

   out = zeros(T)

   n_blocks = T // interval_length
   block_means = mean(s over blocks of size interval_length)

   bins = linspace(0, 1, interval_length)

   for block_idx = 0 to n_blocks-1
       intensity = block_means[block_idx]
       if intensity > 0
           latency_norm = 0.1 * log(1 / intensity)
           latency_norm = clip(latency_norm, 0, 2)

           step = searchsorted(bins, latency_norm)

           t = block_idx * interval_length + clip(step, 0, interval_length-1)
           out[t] = 1
       end if
   end for

   output: out

**Advantages**

- Extremely sparse (one spike per block/neuron) — minimal energy/synaptic events.
- Biologically plausible for rapid sensory processing.
- Robust in population coding (relative latencies preserved despite jitter).

**Disadvantages**

- Limited information per neuron (single spike) — requires population for precision.
- Sensitive to threshold/decay parameters and block length.
- Quantization errors from discrete binning.
- Poor for slowly varying or constant inputs (late/no spike).

For a practical implementation in Python, see the :ref:`Time-to-First-Spike Function <time_to_first_spike_function>`.

**References**

- Rueckauer, et al. (2017). "Conversion of analog to spiking neural networks using sparse temporal coding." *ICAS*. 
- Park, S., et al. (2020). “T2FSNN: deep spiking neural networks with time-to-first-spike coding,” *DAC*.
- Lisman, J. E. (1997). Bursts as a unit of neural information: Making unreliable synapses reliable. *Trends Neurosci*.
