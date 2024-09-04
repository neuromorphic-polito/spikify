.. _filtering_desc:

Filtering
=========

The **filtering** module in the **spikify** library draws inspiration from the intricate processes of the human auditory system, particularly the functioning of the cochlea. This module is designed to mimic the natural filtering mechanisms found in biological systems, offering advanced pre-processing techniques for time-varying signals.

Cochlea-Inspired Filtering
---------------------------

The cochlea, located in the inner ear, performs a frequency decomposition of incoming stimuli. It consists of a spiral structure, the basilar membrane, where nerve cells are arranged in such a way that they function like a filter bank. As the membrane vibrates in response to sound, different regions are excited depending on the frequency of the incoming signal. These excitations are then converted into electrical signals, which are processed by the brain.

In **spikify**, we have implemented a similar approach:

- **Frequency Decomposition**: The filtering techniques break down time-varying signals into multiple frequency components, similar to how the cochlea processes auditory stimuli.
- **Advanced Filters**: Techniques such as Gammatone and Butterworth filters are used to emulate the cochlear filtering process. These filters are effective in splitting the signal into different channels, each corresponding to a specific frequency range.

.. note::

    The **filtering** module is currently under development and will be released in the upcoming version of the **spikify** library. Stay tuned for updates on the release schedule and new features.