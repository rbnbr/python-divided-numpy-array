import math
from itertools import product
import numpy as np
import numba as nb
from src.divarray.divided_array_utils import jit_extents, create_tuple_creator_njit
import time


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
        :param patch_shape:
        """
        # convert to packed numpy array
        array = np.array(obj)

        # self.array = array

        self.shape = array.shape
        self.patch_shape_ = patch_shape  # original patch_shape
        self.patch_shape = tuple(patch_shape[i] if patch_shape[i] is not None else array.shape[i]
                                 for i in range(len(patch_shape)))  # translated to replace None values
        self.dtype = array.dtype

        assert len(patch_shape) == len(self.shape), "patch_shape and shape have to be same length: {} != {}".format(
            len(patch_shape), len(self.shape))
        assert len(self.shape) > 0

        # compute shape of container for the sub arrays
        sub_arrays_shape = tuple(math.ceil(array.shape[i] / self.patch_shape[i]) for i in range(len(array.shape)))

        # divide array into chunks of size patch_shape and fill them into sub_arrays
        self.__sub_arrays__ = DividedArray.__create_subarrays__(sub_arrays_shape, self.patch_shape, array)

        # create array of same size as __sub_arrays__ with their indices as entries
        self.__sub_arrays_indices__ = np.empty_like(self.__sub_arrays__, dtype=np.ndarray)
        # assign indices
        for pos in product(*[range(s) for s in self.__sub_arrays_indices__.shape]):
            self.__sub_arrays_indices__[pos] = pos

        # create accelerated method to fill hypercube with values
        idx_and_patch_to_slices_fn = DividedArray.__create_parse_hypercube_indices_to_patch_slices_func__(
            len(self.shape))
        self.__jit_get_hypercube_patch__ = nb.njit(
            lambda hypercube, idx, shape: hypercube[idx_and_patch_to_slices_fn(idx, shape)]
        )

        shape_to_slice_access_operator_func = DividedArray.__create_shape_to_slice_access_operator_func__(
            len(self.shape))
        list_to_tuple_func = create_tuple_creator_njit(lambda i, l: l[i], len(self.__sub_arrays__.shape))
        self.__jit_get_subarray__ = nb.njit(
            lambda sub_arrays, requested_subarrays_indice, hyper_cube_patch_shape, ps:
            sub_arrays[list_to_tuple_func(requested_subarrays_indice)] if hyper_cube_patch_shape == ps else
            sub_arrays[list_to_tuple_func(requested_subarrays_indice)][shape_to_slice_access_operator_func(
                hyper_cube_patch_shape)]
        )

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
            subarrays_container[pos] = array[obj]

        return subarrays_container

    @staticmethod
    def __parse_hypercube_indices_to_patch_slices__(idx, shape, max_shape):
        return tuple(slice(min(idx[i] * shape[i], max_shape[i]), min((idx[i] + 1) * shape[i], max_shape[i]), None)
                     for i in range(len(shape)))

    @staticmethod
    def __create_parse_hypercube_indices_to_patch_slices_func__(n):
        return create_tuple_creator_njit(lambda i, idx, shape: slice(idx[i] * shape[i], (idx[i] + 1) * shape[i], None),
                                         n)

    @staticmethod
    def __shape_to_slice_access_operator__(shape):
        return tuple(slice(0, s, None) for s in shape)

    @staticmethod
    def __create_shape_to_slice_access_operator_func__(n):
        return create_tuple_creator_njit(lambda i, shape: slice(0, shape[i], None), n)

    @staticmethod
    def __parse_access_int__(i, dim, patch_shape, hypercube_extents=None, isslice=False, slice_step=None, isstart=False):
        if i is None:
            return None
        else:
            if hypercube_extents is None:
                # parse for subarrays
                float_i = i / patch_shape[dim]
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
                            return ret if ret > 0 else None  # specifically for step values less than 0
                        else:
                            return ret
            else:
                ret = i - hypercube_extents[dim][0] * patch_shape[dim]
                if isslice:
                    return ret if ret > 0 else None  # specifically for step values less than 0, this is necessary
                else:
                    return ret

    @staticmethod
    def __parse_access_slice_attr__(i, dim, shape):
        if i is None:
            return None
        elif i < 0:
            return max(0, shape[dim] + i)
        else:
            return min(shape[dim], i)

    @staticmethod
    def __parse_access_slice__(s, dim, shape, patch_shape, hypercube_extents=None):
        if hypercube_extents is None:
            # if hypercube_extens is None, we parse for subarrays
            # for subarray set step always 1 or -1 since we cannot correctly skip full sub arrays
            step = 1 if (s.step is None or s.step > 0) else -1
        else:
            # keep it the same for hypercubes_extents and return
            # don't need to check for collapsed start and stop
            step = 1 if s.step is None else s.step

        start_ = DividedArray.__parse_access_slice_attr__(s.start, dim, shape=shape)
        stop_ = DividedArray.__parse_access_slice_attr__(s.stop, dim, shape=shape)

        start = DividedArray.__parse_access_int__(start_, dim=dim, patch_shape=patch_shape,
                                                  hypercube_extents=hypercube_extents, isslice=True,
                                                  slice_step=step, isstart=True)
        stop = DividedArray.__parse_access_int__(stop_, dim=dim, patch_shape=patch_shape,
                                                 hypercube_extents=hypercube_extents, isslice=True,
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

    @staticmethod
    def __parse_item_to_access_object__(item, shape, patch_shape, dim=None, hypercube_extents=None):
        if isinstance(item, list):
            # parse each item in the list and pass the dim
            return [DividedArray.__parse_item_to_access_object__(i, shape=shape, patch_shape=patch_shape, dim=dim,
                                                                 hypercube_extents=hypercube_extents) for i in item]
        else:
            if dim is None:
                if isinstance(item, int):
                    return DividedArray.__parse_item_to_access_object__(item, shape=shape, patch_shape=patch_shape,
                                                                        dim=0, hypercube_extents=hypercube_extents)
                elif isinstance(item, tuple):
                    # if dim is None and item is a tuple, assign a dim to each item in the tuple and parse it
                    return tuple(DividedArray.__parse_item_to_access_object__(i, shape=shape, patch_shape=patch_shape,
                                                                              dim=d,
                                                                              hypercube_extents=hypercube_extents)
                                 for d, i in enumerate(item))
                elif isinstance(item, slice):
                    return DividedArray.__parse_item_to_access_object__(item, shape=shape, patch_shape=patch_shape,
                                                                        dim=0, hypercube_extents=hypercube_extents)
                else:
                    raise KeyError("type '{}' is not supported by indexing.\ngot element:\n{}".format(type(item), item))
            else:
                if isinstance(item, int):
                    return DividedArray.__parse_access_int__(item, dim=dim, patch_shape=patch_shape,
                                                             hypercube_extents=hypercube_extents)
                elif isinstance(item, tuple):
                    # we got a dim and the item is still a tuple, treat as list
                    return DividedArray.__parse_item_to_access_object__(list(item), shape=shape,
                                                                        patch_shape=patch_shape, dim=dim,
                                                                        hypercube_extents=hypercube_extents)
                elif isinstance(item, slice):
                    return DividedArray.__parse_access_slice__(item, dim=dim, shape=shape, patch_shape=patch_shape,
                                                               hypercube_extents=hypercube_extents)
                else:
                    raise KeyError("type '{}' is not supported by indexing.\ngot element:\n{}".format(type(item), item))

    @staticmethod
    @nb.njit
    def __s_jit_fill_hypercube__(hypercube, sub_arrays, hypercube_indices,
                                 requested_subarrays_indices, patch_shape,
                                 get_hypercube_patch, get_subarray):
        for i in range(len(hypercube_indices)):
            hypercube_patch = get_hypercube_patch(hypercube, hypercube_indices[i], patch_shape)
            hypercube_patch[:] = get_subarray(sub_arrays, requested_subarrays_indices[i],
                                              hypercube_patch.shape, patch_shape)

    def __jit_fill_hypercube__(self, hypercube, sub_arrays, hypercube_indices, requested_subarrays_indices, patch_shape):
        """
        for i in range(len(hypercube_indices)):
            hc_slices = DividedArray.__parse_hypercube_indices_to_patch_slices__(hypercube_indices[i], patch_shape)
            rsi = tuple(requested_subarrays_indices[i])

            hypercube_patch = hypercube[hc_slices]

            if hypercube_patch.shape != patch_shape:
                access_obj = DividedArray.__shape_to_slice_access_operator__(hypercube_patch.shape)
                hypercube_patch[:] = sub_arrays[rsi][access_obj]
            else:
                hypercube_patch[:] = sub_arrays[rsi]
        """
        print(nb.typeof(hypercube))
        print(nb.typeof(sub_arrays))
        print(nb.typeof(hypercube_indices))
        print(nb.typeof(requested_subarrays_indices))
        print(nb.typeof(patch_shape))


        print(self.dtype)
        DividedArray.__s_jit_fill_hypercube__(
            hypercube.astype(np.float64), sub_arrays.astype(np.float64), hypercube_indices, requested_subarrays_indices, patch_shape,
            self.__jit_get_hypercube_patch__, self.__jit_get_subarray__
        )

    def __getitem__(self, item):
        """
        Translates the requested item into the correct access for subarrays and returns the reconstructed result.
        :param item:
        :return:
        """
        shape = self.shape
        patch_shape = self.patch_shape
        sub_arrays_indices = self.__sub_arrays_indices__
        dtype = self.dtype
        sub_arrays = self.__sub_arrays__


        bt = time.time_ns()
        t = time.time_ns()
        # parse item into access object
        # print("pre", item)
        obj = DividedArray.__parse_item_to_access_object__(item, shape, patch_shape)
        # print(obj)
        print("obj parsing took: {}, bt: {}".format(time.time_ns() - t, time.time_ns() - bt))
        t = time.time_ns()

        # get requested subarrays indices
        requested_subarrays_indices = sub_arrays_indices[obj]
        print("getting subarrays took: {}, bt: {}".format(time.time_ns() - t, time.time_ns() - bt))
        t = time.time_ns()

        # if return is a single tuple, map back for consistency
        if type(requested_subarrays_indices) is tuple:
            a = np.empty(shape=(1,), dtype=object)
            a[0] = requested_subarrays_indices
            requested_subarrays_indices = a

        # if shape did collapse, directly return empty array of got shape
        if requested_subarrays_indices.size == 0:
            # TODO: the returned shape is incorrect. I couldn't think of a correct and efficient way to do this a.t.m.
            return np.array([], dtype=dtype).reshape(requested_subarrays_indices.shape)

        print("some checks took: {}, bt: {}".format(time.time_ns() - t, time.time_ns() - bt))
        t = time.time_ns()

        # flatten and remove duplicates
        requested_subarrays_indices = np.unique(requested_subarrays_indices.flatten())

        # convert array of tuple to array of arrays of specific shape
        requested_subarrays_indices = np.asarray(requested_subarrays_indices.tolist()).reshape(
            (len(requested_subarrays_indices), len(shape))
        )

        print("flatting array and cleaning duplicates took: {}, bt: {}".format(time.time_ns() - t, time.time_ns() - bt))
        t = time.time_ns()

        # build shape for hypercube
        hypercube_extents = np.asarray([jit_extents(requested_subarrays_indices[:, i])
                                        if len(requested_subarrays_indices) > 0 else (0, 0)
                                        for i in range(requested_subarrays_indices.shape[1])])

        print("compute shape for hypercube took: {}, bt: {}".format(time.time_ns() - t, time.time_ns() - bt))
        t = time.time_ns()

        # build intermediate hypercube out of requested subarrays indices
        # hypercube = np.empty(tuple(min((hce[2] + 1) * patch_shape[i], shape[i])
        #                            for i, hce in enumerate(hypercube_extents)), dtype=dtype)
        hypercube = np.concatenate(sub_arrays[tuple(requested_subarrays_indices.transpose())])

        print("build empty hypercube of shape {} took: {}, bt: {}".format(hypercube.shape, time.time_ns() - t, time.time_ns() - bt))
        t = time.time_ns()

        # fill hypercube with data from subarrays
        # translates requested subarray indices into hypercube indices
        # hypercube_indices = requested_subarrays_indices - hypercube_extents[:, 0]

        # all_hc_slices = tuple()

        #for i in range(len(hypercube_indices)):
        #    hc_slices = DividedArray.__parse_hypercube_indices_to_patch_slices__(hypercube_indices[i], patch_shape,
        #                                                                         hypercube.shape)
        #    rsi = tuple(requested_subarrays_indices[i])
#
        #    hypercube_patch = hypercube[hc_slices]
#
        #    hypercube_patch[:] = sub_arrays[rsi]
#
        #    # if hypercube_patch.shape != patch_shape:
        #    #     access_obj = DividedArray.__shape_to_slice_access_operator__(hypercube_patch.shape)
        #    #     hypercube_patch[:] = sub_arrays[rsi][access_obj]
        #    # else:
        #    #     hypercube_patch[:] = sub_arrays[rsi]

        print("computing hypercube indices and filling hypercube took: {}, bt: {}"
              .format(time.time_ns() - t, time.time_ns() - bt))
        t = time.time_ns()

        # parse item into access object for resulting hypercube
        hyper_cube_access_obj = DividedArray.__parse_item_to_access_object__(item, shape, patch_shape, dim=None,
                                                                             hypercube_extents=hypercube_extents)
        print("parsing item into access object for hypercube took: {}, bt: {}"
              .format(time.time_ns() - t, time.time_ns() - bt))
        t = time.time_ns()

        # print("hyper_cube_shape", hypercube.shape)
        # print("hyper_cube_access_obj", hyper_cube_access_obj)
        ret = hypercube[hyper_cube_access_obj]
        print("actually accessing hypercube took: {}, bt: {}".format(time.time_ns() - t, time.time_ns() - bt))
        return ret
