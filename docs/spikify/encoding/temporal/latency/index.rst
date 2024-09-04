.. _temporal_latency_desc:

Latency
=================

Latency and inter-spike interval (ISI) are critical components in neural communication, especially in scenarios where the timing between bursts of spikes is crucial for conveying information. When a neuron fires a burst of spikes, the number of spikes (N) can encode information about a specific event. However, the timing between these spikes, known as the inter-spike interval (ISI), also plays a significant role in encoding.

This class of encoding algorithms, known as Latency/ISI, leverages both the timing and number of spikes to improve the reliability and richness of the transmitted information. By incorporating the latency and ISI into the encoding process, these algorithms can capture more nuanced aspects of the input signal, enhancing the neural code's fidelity and precision (Izhikevich et al., 2003).

A prime example of this class is Burst Encoding, which utilizes the burst of spikes and the intervals between them to effectively encode complex information. This approach is particularly advantageous in neural systems where temporal dynamics are key to processing sensory information or triggering specific neural responses.

.. toctree::
   :maxdepth: 1

   burst_encoding