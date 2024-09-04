.. _deconvolution_desc:

Deconvolution
=============

Deconvolution-based encoding techniques are rooted in the mathematical process of reconstructing an analog signal from a spike train. This approach leverages a finite impulse response (FIR) filter to solve the inverse problem, enabling the conversion of analog signals into spikes.

The primary methods in this category include:

- **Hough Spiker Algorithm (HSA)**: Developed by Hough et al. (1999), this algorithm forms the basis of deconvolution-based encoding by translating analog signals into spike trains through convolution operations.

- **Modified-HSA and Ben's Spiker Algorithm (BSA)**: Building upon the HSA, Schrauwen and Van Campenhout (2003) introduced modifications that enhance the robustness and accuracy of the spike generation process. These algorithms refine the spike encoding by adjusting the convolutional filters in a subtractive manner.

Like the Zero-Crossing Step Forward (ZCSF) method, the deconvolution-based techniques produce unipolar spikes, where the presence of spikes indicates positive features of the original signal. This class of algorithms is particularly valuable in scenarios where the precise reconstruction of the original signal's temporal features is critical.

.. toctree::
   :maxdepth: 1

   ben_spiker
   hough_spiker
   modified_hough_spiker