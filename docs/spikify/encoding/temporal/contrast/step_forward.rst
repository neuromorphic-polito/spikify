.. _step_forward_algorithm_desc:

Step Forward (SF) Encoding
=====================

The Step Forward (SF) encoding utilizes an interval around a moving baseline with a set threshold. The initial baseline equals the initial signal value. If the next signal value is above or below baseline ± threshold, a positive or negative spike is registered, respectively, and the baseline is moved to the upper or lower limit of the threshold interval. The threshold is signal amplitude dependent and constitutes the only parameter of this encoding method. The decoding process reconstructs the moving baseline in a manner similar to TBR.

**Algorithm Overview**

SF starts with an initial baseline equal to the first signal value and uses a fixed threshold to decide when to emit spikes. For each subsequent time step, the current signal value is compared to the current baseline ± threshold:

- If the signal exceeds **base + threshold**, a positive spike (+1) is generated and the baseline is updated to **base + threshold**.
- If the signal falls below **base - threshold**, a negative spike (-1) is generated and the baseline is updated to **base - threshold**.
- Otherwise, no spike is emitted and the baseline remains unchanged.

This "step-forward" mechanism ensures that the baseline follows the signal in discrete jumps of size equal to the threshold, providing a staircase-like approximation of the original signal.

**Detailed Pseudocode**

.. code-block:: none
   :linenos:

   SF Encoding Algorithm
   input: s signal, threshold
   base = s(0)
   out = zeros(length(s))
   for t = 1 to length(s)
       if s(t) > base + threshold
           out(t) = 1
           base = base + threshold
       elseif s(t) < base - threshold
           out(t) = -1
           base = base - threshold
       end if
   end for
   output: out, base

**Implementation Steps**:

To implement this algorithm, follow these steps:

1. **Initialize the Base Value** (:math:`\text{base}`): Set the `base` to the first value of the input signal.
2. **Iterate Over Signal**: 

   - For each signal value at time :math:`t`, compare the current signal value to the dynamically updated `base` plus or minus the `threshold`.
   - If the signal exceeds `base + threshold`, generate a positive spike (+1) and update `base` to `base + threshold`.
   - If the signal falls below `base - threshold`, generate a negative spike (-1) and update `base` to `base - threshold`.

**Advantages**:

- Reconstructs most types of continuous signals exceptionally well in both time and frequency domains, including step-wise, smooth, and trended signals.
- Allows multiple steps to account for a single large change, enabling good representation of both small and large amplitude events (depending on threshold choice).
- Preserves the original frequency spectrum without introducing artifact frequency components.
- Introduces only minimal quantization-related noise; does not generate 1/frq ("pink") noise artifacts as seen in TBR.
- The moving baseline prevents drift in the reconstructed signal, even for longer sequences.
- No overshoot occurs, as baseline adjustments are exactly equal to the threshold.
- Exhibits wide, high-fit optimization plateaus across multiple metrics, allowing significant reduction in spike density (lower average firing rate – AFR) without major loss of accuracy.
- Lower AFR enhances data compression, reduces risk of saturation in spiking neural networks (SNN), and can improve noise suppression (though it may magnify quantization noise at low frequencies).
- Robust and straightforward parameter optimization with consistent performance across diverse signal types.

**Disadvantages**

- Reconstruction is inherently stepwise (piecewise constant), which may introduce quantization-like errors, particularly noticeable for very small or gradual changes.
- Small variations below the threshold are completely ignored.
- Like TBR, it encodes only signal changes, so offset errors may occur unless the initial value is correctly matched.
- Noise in the input signal is only minimally reduced by the threshold (mostly passed through rather than filtered).
- The threshold choice remains critical: too large → loss of detail; too small → excessive spikes and higher noise.

For a practical implementation in Python, refer to the :ref:`Step Forward Function <step_forward_function>`.

**References**:

- Kasabov, N., et al. (2016). "Neural Coding Strategies in Spiking Neural Networks." *Neural Processing Letters*.
- Delbruck, T., Lichtsteiner, P. (2007). "Artificial Retina: Applications of Image Processing with Spiking Neural Networks." *IEEE Transactions on Neural Networks*.
- B. Petro, N. Kasabov and R. M. Kiss, "Selection and Optimization of Temporal Spike Encoding Methods for Spiking Neural Networks," in IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 2, pp. 358-370, Feb. 2020, doi: 10.1109/TNNLS.2019.2906158.
