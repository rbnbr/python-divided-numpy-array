from src.divarray.divided_array import DividedArray
import numpy as np
from itertools import product
from tests.access_tester import AccessTester


x = np.empty((54, 144, 144, 3, 8), dtype=object)
# x = np.empty((10, 10), dtype=object)
for pos in product(*[range(s) for s in x.shape]):
    x[pos] = pos

dv = DividedArray(x, (3, 44, 44, 3, None))
# dv = DividedArray(x, (2, 2))


at = AccessTester(x, dv)


def main():
    print(at[0:13:2, 4:14, [1, 2], [0, 1]])
    pass


if __name__ == '__main__':
    main()
