.. _filtering_desc:

Filtering
=========

Preprocessing tools for conditioning time-varying signals before spike encoding.
Frequency decomposition techniques inspired by biological sensory systems break down input signals into multiple frequency channels to improve spike train quality and encoding performance.

Filter Types
------------

- **Gammatone**: Bandpass filterbank approximating the frequency selectivity of the basilar membrane. Suitable for audio and speech signals where cochlear-like decomposition is required.
- **Butterworth**: IIR filterbank with a maximally flat passband response. Suitable for general-purpose signal conditioning and noise attenuation prior to encoding.

Each filter can be configured with parameters such as center frequency, bandwidth, and order. All filters are
implemented as filter banks, decomposing the input signal into multiple frequency channels simultaneously.

For implementation details and usage examples, refer to the :ref:`FilterBank <filterbank_class>` documentation.