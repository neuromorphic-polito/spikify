.. _encoding:

:octicon:`file-directory;1.5em;sd-mr-1 fill-primary` Encoding
=============================================================

Introduction
------------

The **encoding** module in the **spikify** library provides tools for transforming raw data into spikes, a crucial step in spiking neural networks. This module contains two families of algorithms:

- **Rate Coding**: Converts continuous signals into spike trains based on the rate at which spikes occur.
- **Temporal Coding**: Encodes information in the timing of individual spikes, offering a different approach to representing data.

Each family of algorithms offers unique advantages, depending on the nature of the data and the specific requirements of the neural network model.

Algorithm Families
-------------------

### Rate Coding
Rate coding algorithms translate the intensity of the input signal into the frequency of spikes. This approach is often used when the magnitude of the signal is the primary source of information.

### Temporal Coding
Temporal coding algorithms focus on the precise timing of spikes to encode information. This method is particularly effective when the timing of events is more informative than their frequency.

Further Reading
---------------

For detailed explanations and usage guidelines for each coding family, refer to the respective sections:

.. toctree::
   :maxdepth: 1

   rate_coding/index
   temporal_coding/index

These sections dive deeper into each coding approach, providing insights into when and how to use them effectively in your projects.