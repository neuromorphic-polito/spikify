"""GlobalReferenced package."""

from .phase_encoding_algorithm import phase_encoding
from .time_to_spike_algorithm import time_to_first_spike

__all__ = ["phase_encoding", "time_to_first_spike"]
