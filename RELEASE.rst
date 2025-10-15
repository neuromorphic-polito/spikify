Version Release Guidelines
=========================

This document describes the guidelines for releasing new versions of the library.  
We follow semantic versioning, which means our version numbers have three parts: **MAJOR.MINOR.PATCH**.

- **MAJOR** version when you make incompatible API changes
- **MINOR** version when you add functionality in a backwards-compatible manner
- **PATCH** version when you make backwards-compatible bug fixes

Release Steps
-------------


1. **Create a new branch for the release from dev branch:**

   .. code-block:: bash

      git checkout -b x.y.z

2. **Update the version number using the `bump-my-version` command:**

   .. code-block:: bash

      bump-my-version bump patch

   or

   .. code-block:: bash

      bump-my-version bump minor

   or

   .. code-block:: bash

      bump-my-version bump major

3. **Commit the changes with the following message and push the changes to the release branch:**

   .. code-block:: bash

      git commit -m "Bump version: {current_version} → {new_version}"

      git push origin x.y.z

4. **Create a pull request from the release branch to the dev branch.**

5. **Once the pull request is approved and merged, create a new pull request from the dev branch to the master branch.**

6. **Once the pull request is approved and merged, create the tag on the main branch to invoke the package publishing workflow:**

   .. code-block:: bash

      git tag -a x.y.z -m "Release x.y.z"

      git push origin tag <tag_name>

7. **Once the tag is pushed, the package publishing workflow will be triggered and the package will be published to PyPI.**

8. **Once the package is published, create a new release on GitHub with the tag name and the release notes (generate them automatically).**

