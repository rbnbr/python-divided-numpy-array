import numba as nb

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


def main():
    tuple_builder_test()


if __name__ == '__main__':
    main()

