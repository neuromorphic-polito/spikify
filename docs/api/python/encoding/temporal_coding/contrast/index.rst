.. _contrast:

:octicon:`file-directory;1.5em;sd-mr-1 fill-primary` contrast
================================================================

The ``contrast`` folder within the temporal coding module includes algorithms that focus on detecting variations in the signal over time to encode information. These methods are particularly useful for signals where changes or transitions are more informative than steady states.

Contents of the ``contrast`` folder:

- **Moving Window**: Techniques that apply a sliding window to analyze signal variations.
- **Step Forward**: Methods that incrementally evaluate the signal to detect changes.
- **Threshold-Based**: Approaches that trigger spikes based on predefined thresholds.
- **Zero Cross Step Forward**: Algorithms that generate spikes when the signal crosses zero, moving forward in steps.

Below, you will find links to the specific modules for each contrast coding algorithm:

.. toctree::
   :maxdepth: 1

   moving_window
   step_forward
   threshold_based
   zero_cross_step_forward