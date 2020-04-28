"""Excception classes."""


class AstroToolsetNotSpatialError(Exception):
    def __init__(self, array):
        self.array = array

    def __str__(self):
        msg = (
            "Expected a spatial array, but the last dimension has a shape"
            " of %s instead of 3" % self.array.shape[-1]
        )
        return msg
