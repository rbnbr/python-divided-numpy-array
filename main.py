from src.divarray.divided_array import DividedArray
import numpy as np
from itertools import product
from tests.access_tester import AccessTester


x = np.empty((54, 144, 144, 6, 9), dtype=object)
# x = np.empty((10, 10), dtype=object)
for pos in product(*[range(s) for s in x.shape]):
    x[pos] = pos

dv = DividedArray(x, (3, 44, 44, 3, None))
# dv = DividedArray(x, (2, 2))


at = AccessTester(x, dv)


def main():
    _ = at[0:7, 4:8, 15:44, [1, 2]]
    print(at[0:7, 4:33, 15:44, [1, 2]])
    # print(at[(0, slice(None, None, None), )])
    pass


if __name__ == '__main__':
    main()
