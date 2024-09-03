:sd_hide_title:
:hide-toc:

Introduction
============

.. grid::
   :gutter: 2 3 3 3
   :margin: 4 4 1 2
   :class-container: blank-bg
   :class-row: sd-w-100

   .. grid-item::
      :columns: 12 8 8 8
      :child-align: justify
      :class: sd-fs-3

      .. div:: sd-font-weight-bold
         
         Transforming Raw Data into Spiking Neural Signals.

      .. div:: sd-fs-5 sd-font-italic

         Transforming Raw Data into Spiking Neural Signals: Converting Diverse Datasets with Precision for Advanced Neural Network Applications.

      .. grid:: 1 1 2 2
         :gutter: 2 2 3 3
         :margin: 0
         :padding: 0

         .. grid-item::
            :columns: auto

            .. button-ref:: installation
               :ref-type: doc
               :outline:
               :color: primary
               :class: sd-rounded-pill sd-px-4 sd-fs-5

               Get Started

         .. grid-item::
            :columns: auto

            .. button-link:: https://useblocks.com/
               :outline:
               :color: primary
               :class: sd-rounded-pill sd-px-4 sd-fs-5

               About Neuromorphic Polito

   .. grid-item::
      :columns: 12 4 4 4

      .. raw:: html
         :file: _static/process_light.svg

----------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: :octicon:`plug;1.5em;sd-mr-1 fill-primary` Signal-to-Spike Transformation

      Transform raw signals into spikes by choosing from a variety of encoding algorithms tailored to different data characteristics and application needs.

   .. grid-item-card:: :octicon:`gear;1.5em;sd-mr-1 fill-primary` Custom Encoding

      Select and apply your preferred encoding algorithm with ease, giving you full control over how raw signals are converted into spikes.

   .. grid-item-card:: :octicon:`globe;1.5em;sd-mr-1 fill-primary` Advanced Filtering

      Leverage optional filtering inspired by the human cochlea to preprocess your signals, enhancing the quality and relevance of the generated spikes.

   .. grid-item-card:: :octicon:`light-bulb;1.5em;sd-mr-1 fill-primary` SNN Integration

      Directly feed the generated spikes into spiking neural network models, ensuring a smooth and efficient workflow from signal to network.

   .. grid-item-card:: :octicon:`graph;1.5em;sd-mr-1 fill-primary` Future-Ready for NeuroBench

      Planned integration with NeuroBench as a preprocessing algorithm will make “spikify” an essential tool in the SNN research ecosystem.

   .. grid-item-card:: :octicon:`rocket;1.5em;sd-mr-1 fill-primary` High Flexibility

      Whether you’re working with real-time data or offline signals, “spikify” offers high flexibility and performance, making it an ideal choice for diverse spiking neural network applications.

----------------

Contents
--------

.. toctree::
   :caption: Overview
   :maxdepth: 1

   Introduction <self>
   installation
   tutorial

.. toctree::
   :caption: Spikify
   :maxdepth: 1

   spikify/encoding/index
   spikify/filtering/index

.. toctree::
    :caption: API
    :maxdepth: 1

    api/python/index



.. toctree::
   :caption: Development
   :maxdepth: 1

   support
   contributing
   changelog
   licence

* :ref:`genindex`
* :ref:`modindex`
