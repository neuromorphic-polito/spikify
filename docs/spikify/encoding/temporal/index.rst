.. _temporal_coding_desc:

Temporal Coding
===============

**Temporal Coding** is a sophisticated encoding strategy that leverages the timing of spikes to represent information. While traditional Rate Coding focuses on the number of spikes per unit time, Temporal Coding emphasizes the precise timing and intervals between spikes, enabling the encoding of richer and more complex features of the input signal.

This approach is particularly advantageous for capturing fine-grained temporal dynamics, making it ideal for applications where the timing of events carries critical information. Temporal Coding can account for the exact timing of spikes to carry specific information, as well as other temporal characteristics like the relative spike timing and the spacing between spikes.

Within the **spikify** library, Temporal Coding is categorized into several distinct methods:

- **Temporal Contrast**: Exploits the differences in spike timing to enhance contrast in temporal patterns.

- **Latency/ISI**: Utilizes the time intervals between spikes to encode information.

- **Global Referenced**: Considers spikes in the context of global temporal references.

- **Deconvolution-based**: Applies deconvolution techniques to extract temporal information from spike trains.

These methods allow for a versatile and powerful analysis of temporal data, making Temporal Coding a valuable tool in neuromorphic computing and spiking neural networks.

Below, you will find links to the specific description for the temporal coding method:

.. toctree::
   :maxdepth: 1

   contrast/index
   latency/index
   global_referenced/index
   deconvolution/index