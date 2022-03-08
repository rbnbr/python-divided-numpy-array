import numpy as np
import time


class AccessTester:
    """
    Class to test object access via indexing, e.g., obj1[accessor] ?= obj2[accessor].
    Checks if the accessing of the provided first object returns the same as the accessing of the provided
        second object.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __getitem__(self, item):
        res = dict()
        res["item"] = item

        try:
            tns = time.time_ns()
            ar = np.array(self.a[item]).copy()
            a_access_time = time.time_ns() - tns
        except Exception as e:
            print("failed to access first array with error:\n{}".format(e))
            return res

        res["a_access_time"] = a_access_time
        res["ar_size"] = ar.size
        res["ar_shape"] = ar.shape

        try:
            tns = time.time_ns()
            br = np.array(self.b[item]).copy()
            b_access_time = time.time_ns() - tns
        except Exception as e:
            print("failed to access second array with error:\n{}".format(e))
            return res

        res["b_access_time"] = b_access_time
        res["br_size"] = br.size
        res["br_shape"] = br.shape

        # print("accessed both without error. access timings: {}, {}".format(a_access_time, b_access_time))

        # check size
        if ar.size != br.size:
            print("size is not equal: {} != {}. all subsequent tests will fail too.".format(ar.size, br.size))
            return res

        # check shape
        if ar.shape != br.shape:
            print("shape is not equal: {} != {}. subsequent tests may fail too.".format(ar.shape, br.shape))

        # compare values
        try:
            comp_res = np.array(ar == br)
        except Exception as e:
            print("failed to compare values of both arrays with error:\n{}".format(e))
            return res

        res["ar_not_equal_br"] = comp_res.size - comp_res.sum()
        if not comp_res.all():
            print("values are not the same. got {} many unequal values".format(res["ar_not_equal_br"]))
            return res

        return res
