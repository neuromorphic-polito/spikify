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
         
         Transforming Raw Data into Spiking Signals

      .. div:: sd-fs-5 sd-font-italic

         Get ready for neuromorphic computing: convert your data for spiking neural network applications.

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

               Neuromorphic PoliTo

   .. grid-item::
      :columns: 12 4 4 4

      .. raw:: html
         :file: _static/process_light.svg

----------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: :octicon:`webhook;1.5em;sd-mr-1 fill-primary` Signal-to-Spike Transformation

      Transform raw signals into spikes with encoding algorithms tailored to different data characteristics and application needs.

   .. grid-item-card:: :octicon:`gear;1.5em;sd-mr-1 fill-primary` Custom Encoding

      Select the encoding algorithm that best suits your needs and apply it with ease and full control.

   .. grid-item-card:: :octicon:`sliders;1.5em;sd-mr-1 fill-primary` Filtering

      Leverage optional filtering inspired by the human cochlea to preprocess your signals, enhancing the quality and relevance of the generated spikes.

   .. grid-item-card:: :octicon:`command-palette;1.5em;sd-mr-1 fill-primary` SNN Integration

      Directly feed the generated spikes into spiking neural network models, ensuring a smooth and efficient workflow from signal to network.

   .. grid-item-card:: :octicon:`light-bulb;1.5em;sd-mr-1 fill-primary` High Flexibility

      Whether you’re working with real-time or offline data, spikify offers the flexibility you need for your SNN application.

   .. grid-item-card:: :octicon:`rocket;1.5em;sd-mr-1 fill-primary` Try it with NIR and NeuroBench

      Empower your end-to-end design with deployment and benchmarking tools: integrate spikify with NIR and NeuroBench.

----------------

Inspiration
------------

The **spikify**  library is inspired by the research presented in the paper `Spike encoding techniques for IoT time-varying signals benchmarked on a neuromorphic classification task <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.999029/full>`_, which explores various methods for converting continuous signals into spike-based representations for neuromorphic computing.

Contents
--------

.. toctree::
   :caption: Overview
   :maxdepth: 1

   Introduction <self>
   installation
   tutorial

.. toctree::
    :caption: API
    :maxdepth: 1

    api/python/index

.. toctree::
   :caption: Fundamentals
   :maxdepth: 1

   spikify/filtering/index
   spikify/encoding/index

.. toctree::
   :caption: Development
   :maxdepth: 1

   support
   contributing
   changelog

* :ref:`genindex`
* :ref:`modindex`
