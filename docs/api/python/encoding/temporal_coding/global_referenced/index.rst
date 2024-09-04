.. _global_referenced:

:octicon:`file-directory;1.5em;sd-mr-1 fill-primary` global_reference
================================================================

The ``global_referenced`` folder includes algorithms that encode spikes based on global temporal characteristics of the input signal. These methods are particularly useful when timing relative to a global reference, such as an oscillatory signal or stimulus onset, is crucial for encoding information.

Contents of the ``global_referenced`` folder:

- **Phase Encoding**: An algorithm that encodes spikes based on the phase difference with respect to an oscillatory reference.
- **Time-to-First-Spike (TTFS)**: A technique where the time since the onset of the stimulus determines the spike timing, effectively encoding information in the first spike's latency.

Below, you will find links to the specific modules for each global referenced coding algorithm:

.. toctree::
   :maxdepth: 1

   phase_encoding
   time_to_first_spike