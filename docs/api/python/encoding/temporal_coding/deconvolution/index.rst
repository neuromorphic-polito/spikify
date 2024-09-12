.. _deconvolution:

:octicon:`file-directory;0.9em;sd-mr-1 fill-primary` deconvolution
================================================================

The ``deconvolution`` folder within the temporal coding module comprises algorithms that focus on reconstructing analog signals from spike trains. These techniques utilize convolution and inverse convolution operations to encode the signal effectively, often by identifying and subtracting features of the input signal.

Contents of the ``deconvolution`` folder:

- **Ben's Spiker**: A deconvolution-based method that leverages specific patterns in spike trains for signal reconstruction.
- **Hough Spiker**: An algorithm that applies Hough transformation techniques to convert analog signals into spike trains.
- **Modified Hough Spiker**: An adaptation of the Hough Spiker algorithm, offering enhanced performance or additional features.

Below, you will find links to the specific modules for each deconvolution algorithm:

.. toctree::
   :maxdepth: 1

   ben_spiker
   hough_spiker
   modified_hough_spiker