.. _installation:

Installation
============

.. .. only:: html

..    .. image:: https://img.shields.io/pypi/dm/sphinx-needs.svg
..        :target: https://pypi.python.org/pypi/sphinx-needs
..        :alt: Downloads
..    .. image:: https://img.shields.io/pypi/l/sphinx-needs.svg
..        :target: https://pypi.python.org/pypi/sphinx-needs
..        :alt: License
..    .. image:: https://img.shields.io/pypi/pyversions/sphinx-needs.svg
..        :target: https://pypi.python.org/pypi/sphinx-needs
..        :alt: Supported versions
..    .. image:: https://readthedocs.org/projects/sphinx-needs/badge/?version=latest
..        :target: https://readthedocs.org/projects/sphinx-needs/
..        :alt: ReadTheDocs
..    .. image:: https://github.com/useblocks/sphinx-needs/actions/workflows/ci.yaml/badge.svg
..        :target: https://github.com/useblocks/sphinx-needs/actions
..        :alt: GitHub CI Action status
..    .. image:: https://img.shields.io/pypi/v/sphinx-needs.svg
..        :target: https://pypi.python.org/pypi/sphinx-needs
..        :alt: PyPI Package latest release



Using pip
---------
**Spikify** can be installed via `pip`. If you wish to contribute, you can refer to  the :ref:`contributing` section.

To install spikify via `pip`, run the following command:

.. code-block:: bash

    pip install spikify

.. note::

   Consider that the package is still in development, so further updates will be released soon.


Getting code from source
--------------------

To work on spikify from the source code, follow these steps:

.. code-block:: bash

    git clone https://github.com/neuromorphic-polito/spikify
    cd spikify


Library Structure
--------------------

Once installed, you can explore the Spikify library. The general structure of the library is as follows:

.. code-block::

    spikify/
    ├── spikify/
    │   ├── __init__.py
    │   ├── encoding/
    │   │   ├── __init__.py
    │   │   ├── phase_encoding.py
    │   │   ├── rate_encoding.py
    │   ├── filtering/
    │   │   ├── __init__.py
    │   │   ├── low_pass_filter.py
    │   │   ├── high_pass_filter.py
    │   ├── utils/
    │   │   ├── __init__.py
    │   │   ├── validation.py
    │   │   ├── visualization.py
    │   ├── core.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_encoding.py
    │   ├── test_filtering.py
    │   ├── test_utils.py
    ├── examples/
    │   ├── example_script.py
    ├── README.md
    ├── setup.py
    └── requirements.txt

- **spikify/**: The main package directory containing all the core modules and submodules.
  - **encoding/**: Contains encoding algorithms like phase and rate encoding.
  - **filtering/**: Includes various signal filtering methods.
  - **utils/**: Contains utility functions for validation and visualization.
  - **core.py**: The main entry point of the library.
- **tests/**: Directory for unit tests covering different parts of the library.
- **examples/**: Provides example scripts demonstrating how to use Spikify.
- **README.md**: The main documentation file.
- **setup.py**: The setup script for installing the package.
- **requirements.txt**: Lists the package dependencies.

For more detailed information on how to use the various modules and features, please refer to the relevant API documentation:

- :ref:`python_api`
- Additional APIs may be listed here as needed.


