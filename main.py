from src.divarray.divided_array import DividedArray
import numpy as np
from itertools import product
from tests.access_tester import AccessTester


x = np.empty((300, 54, 144, 144, 1, 3), dtype=np.uint8)
# x = np.empty((10, 10), dtype=object)
# for pos in product(*[range(s) for s in x.shape]):
#     x[pos] = pos

dv = DividedArray(x, (1, 3, 44, 44, None, None))
# dv = DividedArray(x, (2, 2))


at = AccessTester(x, dv)


def main():
    # _ = at[5, 0:7, 4:33, 15:44, [1, 2]]
    print(at[0:30, 0:3, 4:33, 15:44])
    # print(at[0:3])
    pass


if __name__ == '__main__':
    main()
