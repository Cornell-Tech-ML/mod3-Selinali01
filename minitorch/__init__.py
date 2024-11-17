"""MiniTorch is a mini deep learning framework focused on tensor computation and automatic differentiation.

This package provides core functionality for:
- Tensor operations and mathematical computations
- Automatic differentiation
- Neural network modules and optimization
- CUDA and fast operation support
- Dataset handling and testing utilities
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .fast_ops import *  # noqa: F401,F403
from .cuda_ops import *  # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from . import fast_ops, cuda_ops  # noqa: F401,F403
