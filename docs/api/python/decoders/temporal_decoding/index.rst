.. _temporal_decoding:

:octicon:`file-directory;0.9em;sd-mr-1 fill-primary` temporal
================================================================

The ``temporal`` module within the decoding module contains algorithms that reconstruct signals based on the timing of individual spikes. This method is especially effective for scenarios where the timing of events conveys more information than their frequency.

Contents of the ``temporal`` module:

- **Contrast**: Algorithms that decode based on variations in the signal over time.
- **Deconvolution**: Approaches that reconstruct the original signal from a spike train using deconvolution techniques.

Below, you will find links to the specific modules for each temporal decoding algorithm:

.. toctree::
   :maxdepth: 1

   contrast/index
   deconvolution/index
