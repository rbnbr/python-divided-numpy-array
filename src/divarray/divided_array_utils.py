import numpy as np
import numba as nb


def create_tuple_creator_njit(f, n):
    # https://github.com/numba/numba/issues/2771#issuecomment-414358902
    assert n > 0
    f = nb.njit(f)
    @nb.njit
    def creator(args):
        return (f(0, *args),)
    for i in range(1, n):
        # need to pass in creator and i to lambda to capture in scope
        @nb.njit
        def creator(args, creator=creator, i=i):
            return creator(args) + (f(i, *args),)
    return nb.njit(lambda *args: creator(args))


def create_tuple_creator_jit(f, n):
    # https://github.com/numba/numba/issues/2771#issuecomment-414358902
    assert n > 0
    f = nb.njit(f)
    @nb.jit
    def creator(args):
        return (f(0, *args),)
    for i in range(1, n):
        # need to pass in creator and i to lambda to capture in scope
        @nb.jit
        def creator(args, creator=creator, i=i):
            return creator(args) + (f(i, *args),)
    return nb.jit(lambda *args: creator(args))


def create_tuple_creator_omjit(f, n):
    # https://github.com/numba/numba/issues/2771#issuecomment-414358902
    assert n > 0
    f = nb.njit(f)
    @nb.jit(nopython=False, forceobj=True)
    def creator(args):
        return (f(0, *args),)
    for i in range(1, n):
        # need to pass in creator and i to lambda to capture in scope
        @nb.jit(nopython=False, forceobj=True)
        def creator(args, creator=creator, i=i):
            return creator(args) + (f(i, *args),)
    return nb.jit(lambda *args: creator(args), nopython=False, forceobj=True)


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

