.. _rate_coding:

:octicon:`file-directory;1.5em;sd-mr-1 fill-primary` rate
================================================

The ``rate`` folder within the encoding module includes algorithms that focus on converting the magnitude or intensity of continuous signals into spike trains based on their rate. This method is particularly useful when the frequency of events is the primary carrier of information in the input signal.

Contents of the ``rate`` folder:

- **Poisson Rate Coding**: Implements a method where the spikes are generated based on the Poisson distribution, providing a stochastic approach to spike generation.

Below, you will find links to the specific modules for each rate coding algorithm:

.. toctree::
   :maxdepth: 1

   poisson_rate