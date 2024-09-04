.. _encoding_desc:

Encoding
========

The encoding techniques  are essential for transforming continuous data into discrete spikes, enabling effective processing in spiking neural networks. With the increasing availability of neuromorphic, event-based sensors, such as silicon retina cameras, the need for efficient spike encoding has become more crucial.

Two primary approaches exist for spike generation:

1. **Model-Based Encoding**: This approach relies on specific neuron models to produce spikes in response to continuous signals, aligning with the Representation Principle of the Neural Engineering Framework (NEF).

2. **Algorithm-Based Encoding**: This approach, which is the focus of the **spikify** library, transforms continuous signals into discrete spikes using a variety of algorithms. This method ensures the full exploitation of neuro-inspired strategies, even in the absence of dedicated neuromorphic hardware.

Classification of Encoding Algorithms
-------------------------------------

The encoding algorithms for spike generation are classified into two main categories:

- **Rate Coding**: Encodes a signal by the number of spikes per time unit, representing the intensity of the input signal.
- **Temporal Coding**: Utilizes the precise timing of spikes to encode information, offering a broader range of approaches depending on the timing of events.

Each algorithm family provides unique advantages, tailored to specific types of input data and application requirements.

Below, you will find links to the specific description for each of these encoding methods:


.. toctree::
   :maxdepth: 1

   rate/index
   temporal/index