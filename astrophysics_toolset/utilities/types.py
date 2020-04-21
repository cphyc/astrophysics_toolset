"""Types used throughout the project"""

from typing import Union, Any
import numpy as np
import pathlib 

PathType = Union[str, pathlib.Path]
IntArrayType = Any # np.ndarray[int]
FloatArrayType = Any # np.ndarray[float]