# This script implements the algorithm Dx-Dy for measuring the quality of a
# two-dimensional self-organizing map (SOM) based on:
# P. Demartines, "Organization measures and representations of the Kohonen
# maps", First IFIP Working Group 10.6 Workshop, 1992.
#
# Copyright (C) 2022  Georgios Is. Detorakis (gdetor@protonmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pylab as plt
import scipy.spatial.distance as ssp
import argparse


def estimateDxDy(X):
    """! Estimates the dy-dx representation of a self-organizing map (SOM).

    @param X Input Numpy array of the feed-forward weights. X has a shape of
    (m, n, z)

    @return The representation dX-dY as dX and dY as well as a line that
    reflects the optimal dX-dY representation (SOMLine)
    """
    # Initialization of parameters
    m, n, z = X.shape[0], X.shape[1], X.shape[2]
    k = m * n
    size = (k**2 - k)//2

    # R contains the coordinates of the regular grid
    R = np.zeros_like(X)

    # In dX are stored the distances of the SOM weights
    dX = np.zeros((size,))
    # In dY are stores the distances of the weights regular grid
    dY = np.zeros((size,))
    for i in range(m):
        for j in range(n):
            R[i, j, 0] = i + 1
            R[i, j, 1] = j + 1

    X = X.reshape(k, z)
    Y = R.reshape(k, z)

    # Calculating all possible distances ((m*n)^2 - (m*n))/2
    dX = ssp.pdist(X, 'euclidean')
    dY = ssp.pdist(Y, 'euclidean')

    # Calculating the som line (dX=dY) this line indicates the perfect matching
    # of dX and dY
    SOMLine = np.arange(size) * dX[0]
    # SOMLine = np.array([[0, 0], [dY.mean(), dX.mean()]])
    return dX, dY, SOMLine


def callEstimateDxDy(filename, n, m, is_plot=True):
    """! This function calls the estimateDxDy function and serves just as an
    example of how one can use the estimateDxDy function.

    @param filename The full path of the feed-forward weights file (str)
    @param n Either the x or y size of the 2D grid of neurons (int)
    @param m The dimension of the feed-forward weights (int)
    @param is_plot If True the function plots the Dx-Dx (bool)

    @return void

    """
    # Load feed-forward weights
    W = np.load(filename)

    # Estimate the dy-dx
    W = W.reshape(n, n, m)
    dX, dY, SOMLine = estimateDxDy(W)

    # Plot the representation Dx-Dy
    if is_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dY, dX, s=1, c='k', marker='s')
        ax.plot(SOMLine, 'r', lw=1.5)
        ax.axis([0, dY.max()+0.15, 0, dX.max()+0.15])
        ax.set_xlabel('dy')
        ax.set_ylabel('dx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate the Dx-Dy measure')
    parser.add_argument('--grid-size-x',
                        type=int,
                        help='dimension of the SOM')
    parser.add_argument('--weights-dim',
                        type=int,
                        help='dimension of the feed-forward weights')
    parser.add_argument('--file',
                        type=str,
                        help='full path of the file that contains the\
                            feed-forward weights')

    args = parser.parse_args()
    callEstimateDxDy(args.file, args.grid_size_x, args.weights_dim)
    plt.show()
