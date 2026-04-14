.. _filtering_desc:

Filtering
=========

The **filtering** module in the **spikify** library is designed to mimic natural filtering mechanisms found in biological systems, offering advanced pre-processing techniques for time-varying signals.

Filtering Techniques
---------------------

In **spikify**, the filtering approach includes:

- **Frequency Decomposition**: The filtering techniques break down time-varying signals into multiple frequency components.
- **Advanced Filters**: Techniques such as Gammatone and Butterworth filters are used to split the signal into different channels, each corresponding to a specific frequency range.

Filter Types Used
------------------

- **Gammatone Filters**: These filters are inspired by the human auditory system and are effective for modeling cochlear frequency selectivity. They are commonly used in audio and speech processing to simulate how biological systems separate sounds.
- **Butterworth Filters**: Known for their maximally flat frequency response in the passband, Butterworth filters are used to cleanly separate signal components without introducing ripples, making them suitable for general-purpose signal filtering.
- **Custom Filter Banks**: The module also supports configurable filter banks, allowing users to tailor the frequency decomposition to specific application needs.

Each filter type can be configured with parameters such as center frequency, bandwidth, and order, providing flexibility for a wide range of signal processing tasks.

For details on how to implement these filters in Python, including examples of configuring Gammatone, Butterworth, and custom filter banks, refer to the :ref:`FilterBank <filterbank_class>` documentation.