from src.divarray.divided_array import DividedArray
import numpy as np
from itertools import product


x = np.empty((54, 144, 144, 3, 8), dtype=object)
# x = np.empty((10, 10), dtype=object)
for pos in product(*[range(s) for s in x.shape]):
    x[pos] = pos

dv = DividedArray(x, (3, 44, 44, 3, None))
# dv = DividedArray(x, (2, 2))

v = False


class SliceParse:
    def __getitem__(self, item):
        xr = np.array(x[item])
        dvr = np.array(dv[item])

        if xr.size == dvr.size == 0:
            print("result is empty, equal value is true")
        else:
            print("xr.shape: {}, dvr.shape: {}".format(xr.shape, dvr.shape))
            print("equal values: {}".format((xr == dvr).all()))

        global v
        if v:
            print(xr)
            print(dvr)
        return item


sp = SliceParse()


def main():
    print(dv[0:13:2, 4:14, [1, 2], [0, 1]])
    pass


if __name__ == '__main__':
    main()
