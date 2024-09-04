.. _encoding:

:octicon:`file-directory;1.5em;sd-mr-1 fill-primary` encoding
=============================================================

The ``encoding`` folder within the spikify library contains the essential components for converting raw data into spike trains, a fundamental step in spiking neural networks. This section is organized to reflect the primary structure of the encoding algorithms included in the library:

- **Rate Coding**: Contains algorithms that convert the intensity of input signals into spike frequency.
- **Temporal Coding**: Encloses algorithms that encode data based on the precise timing of spikes.

Each subfolder is dedicated to a specific family of encoding techniques, making it easy to navigate and understand the purpose of each algorithm within the library structure.

Below, you will find links to the specific modules library for each encoding method:

.. toctree::
   :maxdepth: 1

   rate_coding/index
   temporal_coding/index
