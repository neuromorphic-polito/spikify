Spikify
=======

Spikify is a Python package designed to transform raw signals into spike trains that can be fed into Spiking Neural Networks (SNNs). This package implements a variety of spike encoding techniques based on recent research to facilitate the integration of time-varying signals into neuromorphic computing frameworks.

Introduction
-----------

Spiking Neural Networks (SNNs) are a novel type of artificial neural network that operates using discrete events (spikes) in time, inspired by the behavior of biological neurons. They are characterized by their potential for low energy consumption and computational cost, making them suitable for edge computing and IoT applications. However, traditional digital signals must be encoded into spike trains before they can be processed by SNNs.

This package provides a suite of spike encoding techniques that convert time-varying signals into spikes, enabling seamless integration with neuromorphic computing technologies. The encoding techniques implemented in this package are based on the research article: "Spike Encoding Techniques for IoT Time-Varying Signals Benchmarked on a Neuromorphic Classification Task" (Forno et al., 2022).

Features
--------

* Multiple Spike Encoding Techniques: Includes both rate-based and temporal encoding schemes
* Signal Preprocessing: Tools for preprocessing signals, including Gammatone and Butterworth filters

Installation
-----------

To install the Spikify package, use pip::

    pip install innuce-spikify
