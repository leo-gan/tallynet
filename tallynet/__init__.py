"""TallyNet — tally-coded multi-bit weight networks.

Each parameter is one bit. S bits on one connection are a tally weight; the
count is mapped to the number used in the layer.
"""

from tallynet.encoders import EncoderName, encode_tally
from tallynet.kernels import ComputeName
from tallynet.layers import TallyLinear
from tallynet.models import TallyMLP
from tallynet.packed import pack_pm1, popcount_packed, unpack_pm1

__version__ = "0.1.0"

__all__ = [
    "ComputeName",
    "EncoderName",
    "encode_tally",
    "pack_pm1",
    "popcount_packed",
    "TallyLinear",
    "TallyMLP",
    "unpack_pm1",
    "__version__",
]
