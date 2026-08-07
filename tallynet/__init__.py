"""TallyNet — tally-coded multi-bit weight networks.

Architecture: each matrix entry stores S equal ±1 bits; the tally (count / sum)
is mapped by an encoder to the scalar used in the matmul.
"""

from tallynet.encoders import EncoderName, encode_tally
from tallynet.layers import TallyLinear
from tallynet.models import TallyMLP

__version__ = "0.1.0"

__all__ = [
    "EncoderName",
    "encode_tally",
    "TallyLinear",
    "TallyMLP",
    "__version__",
]
