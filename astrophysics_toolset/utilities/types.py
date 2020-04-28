"""Types used throughout the project"""

from typing import Union, Any
import pathlib

PathType = Union[str, pathlib.Path]
IntArrayType = Any  # np.ndarray[int]
FloatArrayType = Any  # np.ndarray[float]
