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


def main():
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


if __name__ == '__main__':
    main()

