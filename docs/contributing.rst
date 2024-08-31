.. _contributing:

Contributing
============

This page provides a guide for developers wishing to contribute to **spikify**.

Bugs, Features and  PRs
-----------------------

For **bug reports** and well-described **technical feature requests**, please use our issue tracker:
|br| https://github.com/neuromorphic-polito/spikify/issues

For **feature ideas** and **questions**, please use our discussion board:
|br| https://github.com/neuromorphic-polito/spikify/discussions

If you have already created a **PR**, you can send it in. Our CI workflow will check (test and code styles) and
a maintainer will perform a review, before we can merge it.
Your PR and Commit Messages should conform with the following rules:

* A meaningful description or link, which describes the change
* The changed code (for sure :) )
* Test cases for the change (important!)
* Updated documentation, if behavior gets changed or new options/directives are introduced.
* Update of ``docs/changelog.rst``.
* Commit your code following the `Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/>`__ standard.
* If this is your first PR, feel free to add your name in the ``AUTHORS`` file.

In the next sections you will find a detailed guide on how to contribute to the project.

Set up the Project
--------------

1. Fork the project repository to your own GitHub account, clone your fork and configure the remotes.
    .. code-block:: bash

        # Clone the repository from your personal fork into the current directory
        git clone https://github.com/<your-username>/spikify.git
        # Go to the recently cloned folder.
        cd spikify
        # Set the original repository as a remote named "upstream."
        git remote add upstream https://github.com/neuromorphic-polito/spikify.git

2. If it has been some time since you initially cloned, ensure you obtain the most recent updates from the upstream source.:
    .. code-block:: bash

        git checkout <remote-branch-name>
        git pull upstream <remote-branch-name>

3. Create a new branch where you'll develop your feature, change or fix. Name it descriptively to reflect the nature of your work (feat, fix, test, refactor).
    .. code-block:: bash

        git checkout -b <your-branch-name>

Install Dependencies
-----------------------

1. **spikify**  uses `Poetry <https://python-poetry.org/>`__  to manage the project dependencies, so you need to install it first.
    .. code-block:: bash

       pip install poetry
2. Install project dependencies.
    .. code-block:: bash

       poetry install --all-extras

3. Install the `pre-commit <https://pre-commit.com>`__ hooks to run the formatting and linting suite.
    .. code-block:: bash

       pre-commit install

Run Tests
----------------

1. **spikify** use `Pytest <https://docs.pytest.org/en/stable/>`__  to run unit tests on the project and `Codecov <https://about.codecov.io>`__ to generate the coverage report.
    .. code-block:: bash

        # To run unit tests
        poetry run pytest

2. To generate the coverage report, run the following command:
    .. code-block:: bash

        # To generate the coverage report
        poetry run pytest --cov=spikify --cov-report=xml

Build Documentation
-----------------

1. For documentation, follow the `Sphinx docstrings <https://thomas-cokelaer.info/tutorials/sphinx/docstring_python.html>`__  format.
    .. code-block:: bash

        # To build the documentation
        cd docs
        poetry run make html

Linting & Formatting
--------------------

1. **spikify** uses pre-commit hooks to ensure that the code is formatted and linted according to the project's standards.
    .. code-block:: bash

        # To run the pre-commit hooks
        pre-commit run --all-files

.. include:: ../AUTHORS
