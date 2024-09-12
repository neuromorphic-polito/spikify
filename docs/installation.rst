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

Once cloned, you can explore the spikify library. The general structure of the library is as follows:

.. code-block::

    spikify/
    ├── .github/
    ├── docs/
    ├── spikify/
    │   ├── encoding/
    │   │   ├── rate/
    │   │   │   └── ...
    │   │   └── temporal/
    │   │        ├── contrast/
    │   │        │   └── ...
    │   │        ├── deconvolution/
    │   │        │   └── ...
    │   │        ├── global_referenced/
    │   │        │   └── ...
    │   │        └── latency/
    │   │            └── ...
    │   └── filtering/
    │       └── ...
    ├── tests/
    │   ├── encoding/
    │   │   └── ...
    │   └── filtering/
    │       └── ...
    ├── .bumpversion.toml
    ├── .gitignore
    ├── .pre-commit-config.yaml
    ├── AUTHORS
    ├── CITATION.cff
    ├── codecov.yml
    ├── LICENSE
    ├── poetry.lock
    ├── pyproject.toml
    ├── README.rst
    └── RELEASE.rst

**Directory Structure**:

- **.github/**: Contains GitHub-specific configuration files.
- **coverage/**: Stores coverage reports for code testing.
- **docs/**: The documentation folder for Sphinx.
- **spikify/**: The main package directory containing core submodules.
   - **encoding/**: Encoding algorithms for converting raw data into spike trains, organized into:
      - **rate/**: Rate-based encoding methods.
      - **temporal/**: Time-based encoding methods:

        - **contrast/**: Contrast-based encoding algorithms.
        - **deconvolution/**: Deconvolution algorithms.
        - **global_referenced/**: Global reference-based algorithms (e.g., phase encoding).
        - **latency/**: Latency-based encoding algorithms.
   - **filtering/**: Signal filtering methods.
- **tests/**: Contains unit tests to validate the functionality of different parts of the library.
- **.bumpversion.toml**: Configuration file for version bumping using `bumpversion`.
- **.coverage**: File for storing test coverage data.
- **.gitignore**: Specifies files and directories ignored by Git.
- **.pre-commit-config.yaml**: Configuration file for pre-commit hooks.
- **AUTHORS**: A file listing contributors to the project.
- **CITATION.cff**: Citation file for the project.
- **codecov.yml**: Configuration for the Codecov service.
- **LICENSE**: License for the project.
- **poetry.lock**: Lockfile for dependency management with Poetry.
- **pyproject.toml**: Configuration file for Python project tools.
- **README.rst**: The main documentation file providing an overview of the project.
- **RELEASE.rst**: File that lists release notes and changelogs for the project.

For more detailed information on how to use the various modules and features, please refer to the relevant API documentation:

- :ref:`python_api`
- Additional APIs may be listed here as needed.


