import math
from itertools import product
import numpy as np
from src.divarray.divided_array_utils import extents


class DividedArray:
    """
    A divided array simulates a numpy array interface.
    However, the data access is managed differently in the background.
    Provided a given object, a corresponding numpy array is being created.
    This array's memory is not necessarily laid out optimal for the given use case.
    A DividedArray is specifically useful when patches or chunks of the underlying data have to be accessed fast.
    The data is split into chunks of a size which fits nicely into memory.
    This reduced the amount of memory accesses and speeds up the process.
    """
    def __init__(self, obj, patch_shape):
        """
        Constructs a
        :param obj:
        :param patchShape:
        """
        # convert to packed numpy array
        array = np.array(obj)

        # self.array = array

        self.shape = array.shape
        self.patch_shape_ = patch_shape  # original patch_shape
        self.patch_shape = tuple(patch_shape[i] if patch_shape[i] is not None else array.shape[i]
                            for i in range(len(patch_shape))) # translated to replace None values
        self.dtype = array.dtype

        assert len(patch_shape) == len(self.shape), "patch_shape and shape have to be same length: {} != {}".format(
            len(patch_shape), len(self.shape))
        assert len(self.shape) > 0

        # compute shape of container for the sub arrays
        sub_arrays_shape = tuple(math.ceil(array.shape[i] / self.patch_shape[i]) for i in range(len(array.shape)))

        # divide array into chunks of size patch_shape and fill them into sub_arrays
        self.__sub_arrays__ = DividedArray.__create_subarrays__(sub_arrays_shape, self.patch_shape, array)

        # create array of same size as __sub_arrays__ with their indices as entries
        self.__sub_arrays_indices__ = np.empty_like(self.__sub_arrays__, dtype=object)
        # assign indices
        for pos in product(*[range(s) for s in self.__sub_arrays_indices__.shape]):
            self.__sub_arrays_indices__[pos] = pos

    @staticmethod
    def __create_subarrays__(sub_arrays_shape, patch_shape, array):
        """
        returns a subarray containing copies of array.
        :param sub_arrays_shape:
        :param patch_shape:
        :param array:
        :return:
        """
        # create container for subarrays
        subarrays_container = np.empty(sub_arrays_shape, dtype=object)

        for pos in product(*[range(s) for s in subarrays_container.shape]):
            obj = tuple(slice(pos[i] * patch_shape[i], (pos[i] + 1) * patch_shape[i], None)
                        for i in range(len(array.shape)))
            patch = np.empty(patch_shape, dtype=array.dtype)
            sampled_patch = array[obj]
            pobj = DividedArray.__parse_hypercube_indices_to_patch_slice__(np.zeros(len(patch_shape), dtype=int),
                                                                           sampled_patch.shape)
            patch[pobj] = sampled_patch  # this is an assignment to a selection which should act as copy
            subarrays_container[pos] = patch

        return subarrays_container

    @staticmethod
    def __parse_hypercube_indices_to_patch_slice__(idx, shape):
        return tuple(slice(idx[i] * shape[i], (idx[i] + 1) * shape[i], None) for i in range(len(shape)))

    def __parse_access_int__(self, i, dim, hypercube_extents=None, isslice=False, slice_step=None, isstart=False):
        if i is None:
            return None
        else:
            if hypercube_extents is None:
                # parse for subarrays
                float_i = i / self.patch_shape[dim]
                if slice_step is None:
                    return int(float_i)
                elif slice_step > 0:
                    if isstart:
                        return int(float_i)
                    else:
                        return math.ceil(float_i)
                else:
                    if isstart:
                        return math.ceil(float_i)
                    else:
                        ret = int(float_i)
                        if isslice:
                            return ret if ret > 0 else None  # specifically for step values less than 0, this is necessary
                        else:
                            return ret
            else:
                ret = i - hypercube_extents[dim][0] * self.patch_shape[dim]
                if isslice:
                    return ret if ret > 0 else None  # specifically for step values less than 0, this is necessary
                else:
                    return ret

    def __parse_access_slice_attr__(self, i, dim):
        if i is None:
            return None
        elif i < 0:
            return max(0, self.shape[dim] + i)
        else:
            return min(self.shape[dim], i)

    def __parse_access_slice__(self, s, dim, hypercube_extents=None):
        if hypercube_extents is None:
            # if hypercube_extens is None, we parse for subarrays
            # for subarray set step always 1 or -1 since we cannot correctly skip full sub arrays
            step = 1 if (s.step is None or s.step > 0) else -1
        else:
            # keep it the same for hypercubes_extents and return
            # don't need to check for collapsed start and stop
            step = 1 if s.step is None else s.step

        start_ = self.__parse_access_slice_attr__(s.start, dim)
        stop_ = self.__parse_access_slice_attr__(s.stop, dim)

        start = self.__parse_access_int__(start_, dim=dim, hypercube_extents=hypercube_extents, isslice=True,
                                          slice_step=step, isstart=True)
        stop = self.__parse_access_int__(stop_, dim=dim, hypercube_extents=hypercube_extents, isslice=True,
                                         slice_step=step, isstart=False)

        if hypercube_extents is not None:
            return slice(start, stop, step)

        # if start or stop did collapse to the same number, even though they weren't before, we need to fix this
        if start == stop and start is not None:
            if start_ > stop_:
                if step < 0:
                    stop -= 1
                else:
                    # do nothing. if start > stop and step is positive, the result is empty anyways
                    pass
            elif start_ < stop_:
                if step > 0:
                    stop += 1
                else:
                    # do nothing. if start < stop but step is negative, the result is empty anyways
                    pass
            else:
                # do nothing. if start == stop, the result is empty anyways
                pass

        return slice(start, stop, step)

    def __parse_item_to_access_object__(self, item, dim=None, hypercube_extents=None):
        if type(item) is list:
            # parse each item in the list and pass the dim
            return [self.__parse_item_to_access_object__(i, dim=dim, hypercube_extents=hypercube_extents) for i in item]
        else:
            if dim is None:
                if type(item) is int:
                    return self.__parse_item_to_access_object__(item, dim=0, hypercube_extents=hypercube_extents)
                elif type(item) is tuple:
                    # if dim is None and item is a tuple, assign a dim to each item in the tuple and parse it
                    return tuple(self.__parse_item_to_access_object__(i, dim=d, hypercube_extents=hypercube_extents)
                                 for d, i in enumerate(item))
                elif type(item) is slice:
                    return self.__parse_item_to_access_object__(item, dim=0, hypercube_extents=hypercube_extents)
                else:
                    raise KeyError("type '{}' is not supported by indexing.\ngot element:\n{}".format(type(item), item))
            else:
                if type(item) is int:
                    return self.__parse_access_int__(item, dim=dim, hypercube_extents=hypercube_extents)
                elif type(item) is tuple:
                    # we got a dim and the item is still a tuple, treat as list
                    return self.__parse_item_to_access_object__(list(item), dim=dim,
                                                                hypercube_extents=hypercube_extents)
                elif type(item) is slice:
                    return self.__parse_access_slice__(item, dim=dim, hypercube_extents=hypercube_extents)
                else:
                    raise KeyError("type '{}' is not supported by indexing.\ngot element:\n{}".format(type(item), item))

    @staticmethod
    def __shape_to_slice_access_operator__(shape):
        return tuple(slice(0, s, None) for s in shape)

    def __getitem__(self, item):
        """
        Translates the requested item into the correct access for subarrays and returns the reconstructed result.
        :param item:
        :return:
        """
        # parse item into access object
        # print("pre", item)
        obj = self.__parse_item_to_access_object__(item)
        # print(obj)

        # get requested subarrays indices
        requested_subarrays_indices = self.__sub_arrays_indices__[obj]

        # if return is a single tuple, map back for consistency
        if type(requested_subarrays_indices) is tuple:
            a = np.empty(shape=(1,), dtype=object)
            a[0] = requested_subarrays_indices
            requested_subarrays_indices = a

        # if shape did collapse, directly return empty array of got shape
        if requested_subarrays_indices.size == 0:
            # TODO: the returned shape is incorrect. I couldn't think of a correct and efficient way to do this a.t.m.
            return np.array([], dtype=self.dtype).reshape(requested_subarrays_indices.shape)

        # flatten and remove duplicates
        requested_subarrays_indices = np.unique(requested_subarrays_indices.flatten())

        # convert array of tuple to array of arrays of specific shape
        requested_subarrays_indices = np.asarray(requested_subarrays_indices.tolist()).reshape(
            (len(requested_subarrays_indices), len(self.shape))
        )

        # build shape for hypercube
        hypercube_extents = np.asarray([extents(requested_subarrays_indices[:, i], with_difference=True)
                             if len(requested_subarrays_indices) > 0 else (0, 0)
                             for i in range(requested_subarrays_indices.shape[1])])

        # build intermediate hypercube out of requested subarrays indices
        hypercube = np.empty(tuple(min((hce[2] + 1) * self.patch_shape[i], self.shape[i]) for i, hce in enumerate(hypercube_extents)), dtype=self.dtype)

        # fill hypercube with data from subarrays
        # translates requested subarray indices into hypercube indices
        hypercube_indices = requested_subarrays_indices - hypercube_extents[:, 0]

        for i in range(len(hypercube_indices)):
            hc_slice = DividedArray.__parse_hypercube_indices_to_patch_slice__(hypercube_indices[i], self.patch_shape)
            rsi = tuple(requested_subarrays_indices[i])

            hypercube_patch = hypercube[hc_slice]

            if hypercube_patch.shape != self.patch_shape:
                access_obj = DividedArray.__shape_to_slice_access_operator__(hypercube_patch.shape)
                hypercube_patch[:] = self.__sub_arrays__[rsi][access_obj]
            else:
                hypercube_patch[:] = self.__sub_arrays__[rsi]

        # parse item into access object for resulting hypercube
        hyper_cube_access_obj = self.__parse_item_to_access_object__(item, dim=None,
                                                                     hypercube_extents=hypercube_extents)

        # print("hyper_cube_shape", hypercube.shape)
        # print("hyper_cube_access_obj", hyper_cube_access_obj)
        return hypercube[hyper_cube_access_obj]
