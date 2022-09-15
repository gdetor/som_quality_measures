# som_quality_measures

A collection of algorithms for measuring the quality of self-organizing maps.
This repository contains the following implementations:
  1. **Dx-Dy** representation is a relatively simple algorithm for measuring
  the quality of a self-organizing map (SOM). The user can find all the
  details about the Dx-Dy representation in [1].


### Dependencies
  - Numpy
  - Matplotlib
  - Scipy


### Platforms where the code has been tested
  - Ubuntu 20.04.5 LTS
    - GCC 9.4.0
    - Python 3.8.10
    - x86_64


### Example of use

If you'd like to estimate the Dx-Dy representation on your data, you will need 
to store your feed-forward weights in a Numpy file and then use a command like
the one below (do not forget to replace the file name and the rest of the
parameters):

```bash
$ python3 src/som_dxdy.py --grid-size-x 16 --weights-dim 2 --file ./data/weights_sample_noise.npy
```

There are two files, `weights_sample_noise.npy` and `weights_sample_perfect/npy`
in the directory `data/`. You can try out the script by calling it upon those
data files. For the first file, you will see that the Dx-Dy representation is
spreading all over the plane; for the second file, it is aligned over the line
x = y. This means that the first file contains a SOM that's not organized
correctly. On the other hand, the second SOM has perfectly organized and
captured the input space.


### References:
  1. P. Demartines, "Organization measures and representations of the Kohonen
    maps", First IFIP Working Group 10.6 Workshop, 1992.
