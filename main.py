from src.divarray.divided_array import DividedArray
import numpy as np
from itertools import product
from tests.access_tester import AccessTester


x = np.empty((54, 80, 80, 20, 150), dtype=int)
# x = np.empty((10, 10), dtype=object)
# for pos in product(*[range(s) for s in x.shape]):
#     x[pos] = pos

dv = DividedArray(x, (3, 22, 22, None, None))
# dv = DividedArray(x, (2, 2))


at = AccessTester(x, dv)


def main():
    _ = at[0:7, 4:8, 15:44, [1, 2, 5, 18], [4, 5, 6, 26]]
    print(at[0:7, 4:8, 15:44, [1, 2, 5, 18], [4, 5, 6, 26]])
    # print(at[0:3])
    pass


if __name__ == '__main__':
    main()
