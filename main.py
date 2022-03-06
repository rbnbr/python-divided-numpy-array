from src.divarray.divided_array import DividedArray
import numpy as np
from itertools import product
from tests.access_tester import AccessTester


x = np.empty((54, 144, 144, 3, 20), dtype=object)
# x = np.empty((10, 10), dtype=object)
for pos in product(*[range(s) for s in x.shape]):
    x[pos] = pos

dv = DividedArray(x, (3, 44, 44, 3, None))
# dv = DividedArray(x, (2, 2))


at = AccessTester(x, dv)


def main():
    print(at[0:13, 4:30, 15:44])
    pass


if __name__ == '__main__':
    main()
