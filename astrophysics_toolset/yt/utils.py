from unyt.array import  unyt_quantity, unyt_array
def conform_to_qty(ds, value: tuple[float, str] | unyt_quantity) -> unyt_quantity:
    """
    Convert a float-like value to a unyt_quantity.

    Parameters
    ----------
    ds : yt.data_objects.static_output.Dataset
        The dataset to use for unit conversion.
    value : tuple[float, str] | unyt_quantity
        The value to convert.

    Returns
    -------
    unyt_quantity
        The converted value.
    """
    if isinstance(value, unyt_quantity):
        return value

    return ds.quan(value[0], value[1])


def conform_to_arr(ds, value: tuple[float, str] | unyt_array) -> unyt_array:
    """
    Convert a float-like array to a unyt_array.

    Parameters
    ----------
    ds : yt.data_objects.static_output.Dataset
        The dataset to use for unit conversion.
    value : tuple[float, str] | unyt_array
        The value to convert.

    Returns
    -------
    unyt_array
        The converted value.
    """
    if isinstance(value, unyt_array):
        return value

    return ds.arr(value[0], value[1])
