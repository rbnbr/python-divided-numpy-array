import numpy as np
import numba as nb


def extents(a, with_difference=True):
    """
    Returns extents of the given iterable.
    Short for (np.min(a), np.max(a), (if with_difference) max - min)
    :param a:
    :param with_difference:
    :return:
    """
    e = jit_extents(a)
    if with_difference:
        return e
    else:
        return e[0], e[1]


@nb.jit(nopython=False, forceobj=True)
def jit_extents(a):
    if type(a) != np.ndarray:
        a = np.array(a)
    return __extents__(a)


@nb.njit
def __extents__(a):
    mn = np.min(a)
    mx = np.max(a)
    return mn, mx, mx - mn

