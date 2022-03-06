# Subdivided Array

This package is destined benchmark and evaluate different methods for layouting memory in numpy arrays specifically for spatio-temporal data.
The goal is to describe a method, which can access "patches"/chunks of spatio-temporal "close" data faster, than usual numpy arrays do.
Independent of the shape of the numpy array itself, in computer memory, it is laid out as flattened one-dimensional array.
This yields possibly long access times for elements which are close along one axis, because they may be far away in the flattened array.
