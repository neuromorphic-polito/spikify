import numpy as np
import pytest

from spikify.encoders.rate import poisson
from spikify.encoders.temporal.contrast import (
    moving_window,
    step_forward,
    threshold_based_representation,
)
from spikify.encoders.temporal.deconvolution import hough_spiker
from spikify.encoders.temporal.global_referenced import phase, time_to_first_spike
from spikify.encoders.temporal.latency import burst_coding
from spikify.filters import FilterBank

# ---------------------------------------------------------------------------
# Fixtures -- shared test signals
# ---------------------------------------------------------------------------


@pytest.fixture
def signal_1d():
    """A 1-D sinusoidal signal (1000 samples)."""
    np.random.seed(0)
    t = np.linspace(0, 4 * np.pi, 1000)
    return np.sin(2 * t) + 0.5 * np.sin(4 * t)


@pytest.fixture
def signal_2d():
    """A 2-D random signal (500 timestamps x 4 features)."""
    np.random.seed(0)
    return np.random.rand(500, 4)


# ---------------------------------------------------------------------------
# Rate encoding benchmarks
# ---------------------------------------------------------------------------


def test_bench_poisson_1d(benchmark, signal_1d):
    benchmark(poisson, signal_1d, interval_length=10)


def test_bench_poisson_2d(benchmark, signal_2d):
    benchmark(poisson, signal_2d, interval_length=10)


# ---------------------------------------------------------------------------
# Temporal / contrast encoding benchmarks
# ---------------------------------------------------------------------------


def test_bench_threshold_based(benchmark, signal_1d):
    benchmark(threshold_based_representation, signal_1d, factor=0.5)


def test_bench_step_forward(benchmark, signal_1d):
    benchmark(step_forward, signal_1d, threshold=0.2)


def test_bench_moving_window(benchmark, signal_1d):
    benchmark(moving_window, signal_1d, window_length=10, threshold=0.2)


# ---------------------------------------------------------------------------
# Temporal / deconvolution encoding benchmarks
# ---------------------------------------------------------------------------


def test_bench_hough_spiker(benchmark, signal_1d):
    sig = np.abs(signal_1d[:100])  # shorter signal, non-negative
    benchmark(hough_spiker, sig, window_length=5, cutoff=0.1)


# ---------------------------------------------------------------------------
# Global-referenced encoding benchmarks
# ---------------------------------------------------------------------------


def test_bench_phase_encoding(benchmark, signal_1d):
    benchmark(phase, signal_1d, num_bits=8)


def test_bench_time_to_first_spike(benchmark, signal_1d):
    benchmark(time_to_first_spike, signal_1d, interval_length=10)


# ---------------------------------------------------------------------------
# Latency encoding benchmarks
# ---------------------------------------------------------------------------


def test_bench_burst_coding(benchmark, signal_2d):
    benchmark(burst_coding, signal_2d, n_max=4, t_min=2, t_max=6, interval_length=50)


# ---------------------------------------------------------------------------
# Filter bank benchmarks
# ---------------------------------------------------------------------------


def test_bench_filterbank_butterworth(benchmark, signal_1d):
    fb = FilterBank(fs=250, channels=5, f_min=1, f_max=50, order=4, filter_type="butterworth")
    benchmark(fb.decompose, signal_1d)


def test_bench_filterbank_gammatone(benchmark, signal_1d):
    fb = FilterBank(fs=250, channels=5, f_min=1, f_max=50, order=4, filter_type="gammatone")
    benchmark(fb.decompose, signal_1d)
