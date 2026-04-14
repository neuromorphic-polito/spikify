.. _temporal_coding:

:octicon:`file-directory;0.9em;sd-mr-1 fill-primary` temporal
================================================================

The ``temporal`` folder within the encoding module contains algorithms that focus on encoding information based on the timing of individual spikes. This method is especially effective for scenarios where the timing of events conveys more information than their frequency.

Contents of the ``temporal`` folder:

- **Contrast**: Algorithms that encode based on variations in the signal over time.
- **Latency**: Techniques that utilize the latency between spikes (inter-spike interval) for encoding.
- **Global Referenced**: Methods that rely on global temporal characteristics, such as phase encoding or time-to-first-spike.
- **Deconvolution**: Approaches that reconstruct the signal from a spike train using deconvolution techniques.

Below, you will find links to the specific modules for each temporal coding algorithm:

.. toctree::
   :maxdepth: 1

   contrast/index
   latency/index
   global_referenced/index
   deconvolution/index