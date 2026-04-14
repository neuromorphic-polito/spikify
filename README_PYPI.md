<p align="center">
  <img src="https://raw.githubusercontent.com/neuromorphic-polito/spikify/42e47995b238a3dafbfe30480277edde62a2736e/docs/_static/white_logo.svg"
       width="500"
       alt="spikify logo">
</p>

spikify is a Python package designed to transform raw signals into spike trains that can be fed into Spiking Neural Networks (SNNs). This package implements a variety of spike encoding techniques based on recent research to facilitate the integration of time-varying signals into neuromorphic computing frameworks.

## Introduction

Spiking Neural Networks (SNNs) are a novel type of artificial neural network that operates using discrete events (spikes) in time, inspired by the behavior of biological neurons. They are characterized by their potential for low energy consumption and computational cost, making them suitable for edge computing and IoT applications. However, traditional digital signals must be encoded into spike trains before they can be processed by SNNs.

This package provides a suite of spike encoding techniques that convert time-varying signals into spikes, enabling seamless integration with neuromorphic computing technologies. The encoding techniques implemented in this package are based on the research article: "Spike Encoding Techniques for IoT Time-Varying Signals Benchmarked on a Neuromorphic Classification Task" (Forno et al., 2022).

## Installation

To install the spikify package, use pip:

```bash
pip install innuce-spikify
```

## Usage

Here is a simple example to get started:

```python
import numpy as np
from spikify.filtering import FilterBank
from spikify.encoding.rate import poisson

# Generate a sinusoidal signal
time = np.linspace(0, 4 * np.pi, 200)
signal = np.sin(2 * time) + 0.5 * np.sin(4 * time)

filter = FilterBank(fs=50, channels=5, f_min=0.5, f_max=5, order=4, filter_type='butterworth')

filtered_signal = filter.decompose(signal) # (timesteps, channels, features)

filtered_signal = np.reshape(filtered_signal, (-1, filtered_signal.shape[1] * filtered_signal.shape[2]))

# Encode the filtered signal
encoded_signal = poisson(filtered_signal, interval_length=2)
```

For more detailed examples and usage, please refer to the [documentation](https://spikify.readthedocs.io/en/latest/).

## Encoding Techniques

spikify implements the following spike encoding families:

| Encoding Family         | Algorithm                | Description                                                                                       |
|------------------------|--------------------------|---------------------------------------------------------------------------------------------------|
| **Rate Encoding**      | Poisson Rate             | Models spike generation as a Poisson process; instantaneous firing rate proportional to signal amplitude |
| **Temporal Encoding**  | Threshold-Based          | Fires an ON spike when the signal crosses a positive threshold, and an OFF spike when it crosses a negative one                           |
|                        | Step Forward             | Fires ON or OFF spikes each time the signal accumulates enough change in either direction                                    |
|                        | Zero-Cross Step Forward  | Simplified version of the step-forward that encodes only positive signals                               |
|                        | Moving Window            | Fires positive or negative spikes when the signal rises or drops significantly within a short local window                              |
| **Deconvolution-Based**     | Hough Spiker             | Implements an iterative subtraction procedure between the signal and a convolution filter                        |
|                        | Modified Hough Spiker    | Extends Hough Spiker with outlier rejection for noise-robust encoding                            |
|                        | Bens Spiker               | Extends the Hough Spiker with an additional error control threshold                       |
| **Global Referenced**  | Phase Encoding           | Use the inverse arcsin transformation of the signal to compute the binary pattern based on a quantized local mean value of the input                  |
|                        | Time-to-First Spike            | Encodes amplitude as latency delay from stimulus onset to first spike                            |
| **Latency Encoding**   | Burst Coding             | Represents signal intensity via inter-spike interval within a burst                              |

**Tips:**  
- Use **Poisson Rate** for general-purpose encoding.  
- Use **Temporal** or **Deconvolution** methods for signals where timing or event structure is important.

## Filters

spikify provides preprocessing filters to condition raw signals before encoding. Both filters are implemented as filter banks with configurable channels, frequency bounds, and order.

| Filter Type      | Description                                                                                          |
|-----------------|------------------------------------------------------------------------------------------------------|
| **Gammatone**   | Bandpass filterbank approximating basilar membrane response; models cochlear frequency decomposition |
| **Butterworth** | IIR low-pass filter with maximally flat passband; attenuates high-frequency noise before encoding   |

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