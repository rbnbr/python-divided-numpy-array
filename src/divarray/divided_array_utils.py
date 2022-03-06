import numpy as np
# from numba import jit


def extents(a, with_difference=False, *, axis=None, keepdims=np._NoValue, initial_min=np._NoValue, initial_max=np._NoValue,
            out_min=None, out_max=None, where_min=np._NoValue, where_max=np._NoValue):
    """
    Returns extents of the given iterable.
    Short for (np.min(a), np.max(a), (if with_difference) max - min)
    :param a:
    :return:
    """
    _min = np.min(a, axis=axis, out=out_min, keepdims=keepdims, initial=initial_min, where=where_min)
    _max = np.max(a, axis=axis, out=out_max, keepdims=keepdims, initial=initial_max, where=where_max)
    if with_difference:
        return _min, _max, _max - _min
    else:
        return _min, _max


# @jit
# def extents(a, with_difference=False):
#     """
#     Returns extents of the given iterable.
#     Short for (np.min(a), np.max(a), (if with_difference) max - min)
#     :param a:
#     :param with_difference:
#     :return:
#     """
#     _min = np.min(a)
#     _max = np.max(a)
#     if with_difference:
#         return _min, _max, _max - _min
#     else:
#         return _min, _max, 0

