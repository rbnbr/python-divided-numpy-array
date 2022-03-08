# Subdivided Array

**check the Note below!**

This package is destined benchmark and evaluate different methods for layouting memory in numpy arrays specifically for spatio-temporal data.
The goal is to describe a method, which can access "patches"/chunks of spatio-temporal "close" data faster, than usual numpy arrays do.
Independent of the shape of the numpy array itself, in computer memory, it is laid out as flattened one-dimensional array.
This yields possibly long access times for elements which are close along one axis, because they may be far away in the flattened array.

## Note
First tests seems to show that there is now relevant speed up compared to the standard numpy access operator.
However, this approach could still be useful to deal with huge datasets which cannot be loaded into ram at once but need to be read from disk.
The implementation needs multiple adjustments to make this work.

Nevertheless, if you are interested on how to adjust the access item for a numpy array to access specific chunks in a downscales array, then check the \__getitem\__ implementation of the DividedArray.


### TODOs
- Enable the implementation to work with chunks saved to disk.
- Allow to write to the data in subarrays via the returned object.

