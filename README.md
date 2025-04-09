# Spikify

Spikify is a Python package designed to transform raw signals into spike trains that can be fed into Spiking Neural Networks (SNNs). This package implements a variety of spike encoding techniques based on recent research to facilitate the integration of time-varying signals into neuromorphic computing frameworks.

## Introduction

Spiking Neural Networks (SNNs) are a novel type of artificial neural network that operates using discrete events (spikes) in time, inspired by the behavior of biological neurons. They are characterized by their potential for low energy consumption and computational cost, making them suitable for edge computing and IoT applications. However, traditional digital signals must be encoded into spike trains before they can be processed by SNNs.

This package provides a suite of spike encoding techniques that convert time-varying signals into spikes, enabling seamless integration with neuromorphic computing technologies. The encoding techniques implemented in this package are based on the research article: "Spike Encoding Techniques for IoT Time-Varying Signals Benchmarked on a Neuromorphic Classification Task" (Forno et al., 2022).

## Features

* Multiple Spike Encoding Techniques: Includes both rate-based and temporal encoding schemes
* Signal Preprocessing: Tools for preprocessing signals, including Gammatone and Butterworth filters

## Installation

To install the Spikify package, use pip:

```bash
pip install innuce-spikify
```

## Usage

Here is a simple example to get started:

```python
import numpy as np

# Generate a sinusoidal signal
time = np.linspace(0, 2 * np.pi, 100)  # Time from 0 to 2*pi
amplitude = np.sin(time)  # Sinusoidal signal

# Encode the raw signal into a spike train using Poisson Rate Coding
from spikify.encoding.rate import poisson_rate

# Set parameters for encoding
np.random.seed(0)  # For reproducibility
interval_length = 2  # Length of the encoding interval

# Encode the sinusoidal signal
encoded_signal = poisson_rate(amplitude, interval_length)
```

For more detailed examples and usage, please refer to the [documentation](https://spikify.readthedocs.io/en/latest/).

## Encoding Techniques

This package implements several spike encoding families techniques, including:

### Rate Encoding

Rate encoding represents information by the firing rate of neurons. The higher the stimulus intensity, the higher the firing rate.

Algorithms:
* Poisson Rate

### Temporal Encoding

Temporal encoding conveys information through the precise timing of spikes. This family contains subcategories for contrast and deconvolution techniques:

#### Contrast-Based Temporal Encoding

Algorithms:
* Moving Window
* Step Forward
* Threshold-Based
* Zero-Cross Step Forward

#### Deconvolution-Based Temporal Encoding

Algorithms:
* Ben Spiker
* Hough Spiker
* Modified Hough Spiker

#### Global Referenced Encoding

Algorithms:
* Phase Encoding
* Time-to-Spike

#### Latency Encoding

Algorithms:
* Burst Encoding

Each technique has its advantages and can be selected based on the type of input data and the desired SNN architecture.

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