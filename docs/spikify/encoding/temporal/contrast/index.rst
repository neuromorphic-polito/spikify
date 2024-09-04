.. _temporal_contrast_desc:

Contrast
========

Temporal Contrast algorithms are designed to capture and encode variations in the signal over time. These methods are particularly effective in detecting changes, whether positive or negative, in the input signal, leading to the production of spikes that reflect these temporal variations. Due to their focus on time-based dynamics, these algorithms are less suited for static spatial data, such as still images, where temporal change is absent.

The primary use cases for Temporal Contrast algorithms involve data types where signal variation over time is crucial. This includes:

- **Audio Signals**: Detecting pitch, tone, or other temporal features in sound data (Liu et al., 2014).

- **Electromyography (EMG) Data**: Capturing muscle activity through electrical signals over time (Donati et al., 2019).

- **Speech Recognition**: Identifying spoken words or phrases based on temporal patterns (Gao et al., 2019).

- **Failure Prediction in Machines**: Monitoring vibrations or other operational signals to predict breakdowns (Dennler et al., 2021).

- **Robotic Braille Reading**: Enabling robots to read Braille by detecting the temporal changes in the tactile input (Müller-Cleve et al., 2022).


These use cases highlight the versatility and effectiveness of Temporal Contrast algorithms in real-world applications.

.. toctree::
   :maxdepth: 1

   moving_window
   step_forward
   threshold_based
   zero_cross_step_forward
