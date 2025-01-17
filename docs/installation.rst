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
**spikify** can be installed via `pip`. If you wish to contribute, you can refer to  the :ref:`contributing` section.

To install spikify via `pip`, run the following command:

.. code-block:: bash

    pip install spikify

.. note::

   Consider that the package is still in development, so further updates will be released soon.


System Requirements
-------------------
- Python >= 3.10
- OS: Compatible with Linux, macOS, and Windows

Using a Virtual Environment
---------------------------
It's recommended to install spikify in a virtual environment to avoid conflicts with other Python packages. You can create a virtual environment with the following commands:

.. code-block:: bash

   python -m venv spikify-env
   source spikify-env/bin/activate  # On Windows use `spikify-env\Scripts\activate`
   pip install spikify

Getting code from source
------------------------

To work on spikify from the source code, follow these steps:

.. code-block:: bash

   git clone https://github.com/neuromorphic-polito/spikify
   cd spikify

Verifying Installation
-----------------------
To verify that spikify was installed correctly, try importing it in Python:

.. code-block:: python

   import spikify
   print(spikify.__version__)


Troubleshooting
---------------
- If you encounter issues related to permissions, try installing with `--user`:

.. code-block:: bash

  pip install --user spikify

- For issues related to dependencies or versions, make sure your pip and setuptools are up to date:

.. code-block:: bash

  pip install --upgrade pip setuptools


For more detailed information on how to use the various modules and features, please refer to the relevant API documentation:

- :ref:`python_api`
- Additional APIs may be listed here as needed.
