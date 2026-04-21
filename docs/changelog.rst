.. _changelog:

Changelog
===================

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`__,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`__.

1.1.0
-----
:Released: 2026-04-21
:Full Changelog: `1.0.0...1.1.0 <https://github.com/neuromorphic-polito/spikify/compare/1.0.0...1.1.0>`__

Improvements
............
- Introduced decoding support for contrast-based and deconvolution-based encoders, enabling signal reconstruction workflows.
- Improved return format consistency by removing implicit flattening of outputs across decoding functions.
- Clarified argument order and improved API consistency in decoding-related functions.
- General documentation style improvements and cleanup.
- Removed unnecessary argument initializations to simplify internal logic.

1.0.0
-----
:Released: 2026-04-14
:Full Changelog: `0.3.0...1.0.0 <https://github.com/neuromorphic-polito/spikify/compare/0.3.0...1.0.0>`__

Improvements
............
- Renamed ``filtering`` and ``encoding`` packages to ``filters`` and ``encoders`` for improved consistency and shorter import paths.
- Renamed ``burst encoding`` to ``burst coding`` to shorten the module path and align with conventional terminology.
- Renamed the global-referenced phase encoding and Poisson encoding functions for clarity and consistency.
- Enhanced ``FilterBank`` to enforce 2D signal processing, improve filter coefficient selection, and expand test coverage on center frequencies.
- Added automatic signal scaling and normalization to several encoding algorithms.
- Added FIR filter support to the BSA encoder along with updated parameters and documentation.
- Extended ``codemeta.json`` with auto-updating version, download link, and date fields.
- Added a PyPI-compatible README with updated logo and layout.

Bug Fixes
.........
- Fixed the SF algorithm timestep calculation.
- Fixed the MW algorithm implementation.
- Fixed the HSA issue affecting spike generation correctness.
- Fixed the TBR output by replacing variation prepend with append to match the original algorithm setup.
- Fixed the ``codemeta.json`` version identifier.

Documentation
.............
- Refactored and expanded documentation for TBR, BSA, HSA, SF, MW, ZCSF, burst coding, phase encoding, Poisson encoding, and contrast-based algorithms, including pseudocode and improved descriptions.
- Renamed documentation paths to reflect the new ``filters`` and ``encoders`` module structure.
- Clarified descriptions for global-referenced encoding algorithms and filter sections.

0.3.0
-----
:Released: 2025-11-12
:Full Changelog: `0.2.0...0.3.0 <https://github.com/neuromorphic-polito/spikify/compare/0.2.0...0.3.0>`__

Improvements
............

- Introduced a new `FilterBank` class that allows users to create and apply filter banks (e.g., Gammatone, Butterworth) to decompose input signals into multiple frequency channels.
- Enhanced documentation with additional examples and usage guidelines for the new filtering capabilities.


0.2.0
-----
:Released: 2025-10-15
:Full Changelog: `0.1.1...0.2.0 <https://github.com/neuromorphic-polito/spikify/compare/0.1.1...0.2.0>`__

Improvements
............

- Added multiple features encoding for every encoding method, allowing users to encode multiple features simultaneously.



0.1.1
-----
:Released: 2025-05-22
:Full Changelog: `0.1.0...0.1.1 <https://github.com/neuromorphic-polito/spikify/compare/0.1.0...0.1.1>`__

Bug Fixes
...........

- Fix bug in the `phase_encoding` function that caused incorrect spike generation under certain conditions. This was due to an erron on how bits were unpacked.



0.1.0
-----

:Released: 2022-03-01
:Full Changelog: `0.1.0...HEAD <https://github.com/neuromorphic-polito/spikify/compare/0.1.0...HEAD>`__

.. Improvements
.. ............



.. Bug fixes
.. .........


.. Internal improvements
.. ......................

