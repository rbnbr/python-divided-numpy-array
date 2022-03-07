import numba as nb
from itertools import product
import numpy as np
import time


@nb.jit(nopython=False, forceobj=True)
def extents(a):
    if type(a) != np.ndarray:
        a = np.array(a)
    return __extents__(a)


@nb.njit
def __extents__(a):
    mn = np.min(a)
    mx = np.max(a)
    return mx, mn, mx - mn


def nojit_extents(a):
    mn = np.min(a)
    mx = np.max(a)
    return mx, mn, mx - mn


def extents_wd(a, with_difference=True):
    """
    Returns extents of the given iterable.
    Short for (np.min(a), np.max(a), (if with_difference) max - min)
    :param a:
    :param with_difference:
    :return:
    """
    e = extents(a)
    if with_difference:
        return e
    else:
        return e[0], e[1]


def test_f(n, e_fn):
    a = [np.arange((i + 1)) for i in range(n)]
    st = 0
    for i in range(n):
        t = time.time_ns()
        e = e_fn(a[i])
        st += time.time_ns() - t
    return st


def test_extents():
    tests = [0, 1, -2, (0, 1), 0.1, -0.2, (0, 0.1, -2, -2.2),
             np.array(0), np.array((-1, 2, -1)), np.array((0.1, -0.2, 5))]

    for t in tests:
        _ = (extents(t), nojit_extents(t))

    print("\nstart test")
    n = 50000
    print("n={}".format(n))
    print("pure jit: {}".format(test_f(n, __extents__)))
    print("mixed jit: {}".format(test_f(n, extents)))
    print("no jit: {}".format(test_f(n, extents_wd)))


def create_tuple_creator(f, n):
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


@nb.njit
def tuple_add_value(t: tuple, b):
    return t + (b, )


@nb.njit
def tuple_builder(x):
    t = ()
    for i in range(x):
       t = tuple_add_value(t, i)
    return t


@nb.jit(nopython=False, forceobj=True)
def __shape_dim_to_slice_access_operator__(shape_dim):
    return slice(0, shape_dim, None)


def var_range(x):
    range_x = create_tuple_creator(lambda i: i, x)
    @nb.njit
    def foo():
        print(range_x())
    return foo


def tuple_builder_test():
    # print(tuple_builder(0))
    # print(tuple_add_value((), 1))
    print(__shape_dim_to_slice_access_operator__(0))


@nb.njit
def slice_test(a, start, stop, step):
    s = slice(start, stop, step)
    return a[s]


# accelerate this
def __shape_to_slices_access_operator__(shape):
    return tuple(slice(0, s, None) for s in shape)


def __shape_to_slice_access_operator__(s):
    return slice(0, s, None)


def tp_creator(tp_len):
    # call in init depending on shape length
    tp = create_tuple_creator(lambda i, shape: slice(0, shape[i], None), tp_len)
    return tp


@nb.njit
def use_tuple_slice_builder_in_njit(a, shape):
    tp = tp_creator(len(shape))
    return a[tp(shape)]


def tuple_creator_example(n):
    return create_tuple_creator(lambda i, idx, shape: slice(idx[i], idx[i] + shape[i], None), n)


@nb.njit
def njit_accessor(xr, s_ret):
    return xr[s_ret]


def print_tce(idx, shape, xr):
    tce = tuple_creator_example(len(shape))
    nb.njit(lambda x: print(njit_accessor(x, tce(idx, shape))))(xr)


@nb.njit
def njit_type(a):
    if isinstance(a, tuple):
        return True
    else:
        return False


@nb.njit
def _jit_list_of_objects_(l, access):
    return l[access[0]][access[1]]


def list_of_objects(shape, access):
    x = np.empty(shape, dtype=object)
    for pos in product(*[range(s) for s in x.shape]):
        x[pos] = sum(pos)
    print(nb.typeof(x.tolist()))
    return _jit_list_of_objects_(x, access)


def main():
    # print(slice_test(np.arange(10), 0, None, 1))
    # shape = (5, 10, 5, 10)
    # print(use_tuple_slice_builder_in_njit(np.empty(shape), shape))
    # print_tce((5, 6, 7), (1, 2, 3), np.empty((10, 10, 10)))
    # print(njit_type(list()))
    list_of_objects((10, 10), (0, 1))


if __name__ == '__main__':
    main()

