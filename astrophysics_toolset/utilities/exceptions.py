"""Excception classes."""


class AstroToolsetNotSpatialError(Exception):
    def __init__(self, array):
        self.array = array

    def __str__(self):
        msg = (
            "Expected a spatial array, but the last dimension has a shape"
            f" of {self.array.shape[-1]} instead of 3"
        )
        return msg
