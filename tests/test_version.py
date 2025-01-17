import unittest
from spikify import __version__


class TestVersion(unittest.TestCase):

    def test_version(self):
        assert __version__ is not None
