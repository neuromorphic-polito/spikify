.. _decoding:

:octicon:`file-directory;0.9em;sd-mr-1 fill-primary` decoders
=============================================================

The ``decoders`` module within the spikify library contains the essential components for converting spike trains back into continuous signals or meaningful representations. This section is organized to reflect the primary structure of the decoding algorithms included in the library:

- **Temporal Coding**: Encloses algorithms that decode data based on the precise timing of spikes, reconstructing signals from spike timing information.

Each submodule is dedicated to a specific family of decoding techniques, making it easy to navigate and understand the purpose of each algorithm within the library structure.

Below, you will find links to the specific modules library for each decoding method:

.. toctree::
   :maxdepth: 1

   temporal_decoding/index
