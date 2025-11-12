<p align="center">
  <img src="https://github.com/neuromorphic-polito/spikify/blob/main/docs/_static/white_logo.svg" alt="Spikify Overview" />
</p>

Spikify is a Python package designed to transform raw signals into spike trains that can be fed into Spiking Neural Networks (SNNs). This package implements a variety of spike encoding techniques based on recent research to facilitate the integration of time-varying signals into neuromorphic computing frameworks.

## Introduction

Spiking Neural Networks (SNNs) are a novel type of artificial neural network that operates using discrete events (spikes) in time, inspired by the behavior of biological neurons. They are characterized by their potential for low energy consumption and computational cost, making them suitable for edge computing and IoT applications. However, traditional digital signals must be encoded into spike trains before they can be processed by SNNs.

This package provides a suite of spike encoding techniques that convert time-varying signals into spikes, enabling seamless integration with neuromorphic computing technologies. The encoding techniques implemented in this package are based on the research article: "Spike Encoding Techniques for IoT Time-Varying Signals Benchmarked on a Neuromorphic Classification Task" (Forno et al., 2022).

## Features

* Multiple Spike Encoding Techniques: Includes both rate-based and temporal encoding schemes
* **Signal Preprocessing:** Tools for filtering and preparing signals, including:
  * **Gammatone Filter:** Mimics human auditory filtering, useful for audio and speech signals.
  * **Butterworth Filter:** Smooths and removes noise from signals, ideal for general sensor data.
  * Easily chain filters before encoding to improve spike train quality.

## Installation

To install the Spikify package, use pip:

```bash
pip install innuce-spikify
```

## Usage

Here is a simple example to get started:

```python
import numpy as np
from spikify.filtering import FilterBank
from spikify.encoding.rate import poisson_rate

# Generate a sinusoidal signal
time = np.linspace(0, 2 * np.pi, 100)  # Time from 0 to 2*pi
amplitude = np.sin(time)  # Sinusoidal signal

filter = FilterBank(fs=50, channels=5, f_min=0.5, f_max=5, order=4, filter_type='butterworth')

filtered_signal = filter.decompose(signal) # (timesteps, channels, features)

filtered_signal = np.reshape(filtered_signal, (-1, filtered_signal.shape[1] * filtered_signal.shape[2]))

# Encode the filtered signal
encoded_signal = poisson_rate(filtered_signal, interval_length=2)
```

For more detailed examples and usage, please refer to the [documentation](https://spikify.readthedocs.io/en/latest/).

## Encoding Techniques

This package implements several spike encoding families techniques, including:

| Encoding Family         | Algorithm                | Description                         |
|------------------------|--------------------------|--------------------------------------|
| **Rate Encoding**      | Poisson Rate             | Encodes intensity as firing rate     |
| **Temporal Encoding**  | Moving Window            | Spikes on local changes              |
|                        | Step Forward             | Spikes on signal steps               |
|                        | Threshold-Based          | Spikes when crossing thresholds      |
|                        | Zero-Cross Step Forward  | Spikes on zero-crossings             |
| **Deconvolution-Based**| Ben Spiker               | Deconvolves spikes from signal       |
|                        | Hough Spiker             | Uses Hough transform for spikes      |
|                        | Modified Hough Spiker    | Robust Hough-based encoding          |
| **Global Referenced**  | Phase Encoding           | Encodes phase information            |
|                        | Time-to-Spike            | Spikes at specific time delays       |
| **Latency Encoding**   | Burst Encoding           | Encodes bursts of spikes             |

**Tip:**  
- Use **Poisson Rate** for general-purpose encoding.  
- Use **Temporal** or **Deconvolution** methods for signals where timing or event structure is important.

## Filters

Spikify provides preprocessing filters that can be applied to signals before encoding to improve spike train quality and remove noise. These filters help condition the raw signal data for better encoding performance.

### Available Filters

| Filter Type        | Description                                        |
|-------------------|----------------------------------------------------|
| **Gammatone**     | Mimics human auditory filtering                   |
| **Butterworth**   | Low-pass filter for noise reduction and smoothing |


## Encoded Datasets

The following datasets have been selected to serve as examples for benchmarking spike train encoding techniques:

* WISDM Dataset: 20 Hz recordings of human activity through mobile and wearable inertial sensors

These datasets are preprocessed and converted into spike trains to evaluate the performance of different encoding techniques.

## Citation

If you use this framework in your research, please cite the following article:

```bibtex
@ARTICLE{
    10.3389/fnins.2022.999029,
    AUTHOR={Forno, Evelina  and Fra, Vittorio  and Pignari, Riccardo  and Macii, Enrico  and Urgese, Gianvito },
    TITLE={Spike encoding techniques for IoT time-varying signals benchmarked on a neuromorphic classification task},
    JOURNAL={Frontiers in Neuroscience},
    VOLUME={16},
    YEAR={2022},
    URL={https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.999029},
    DOI={10.3389/fnins.2022.999029},
    ISSN={1662-453X},
}
```

## Contributing

We welcome contributions from the community. Please see our CONTRIBUTING.rst file for more details on how to get involved.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details. 