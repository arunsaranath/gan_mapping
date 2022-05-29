# -*- coding: utf-8 -*-

"""
FileName:               generalUtilities
Author Name:            Arun M Saranathan
Description:            This file includes implementation of general utility function which are used for CRISM data
                        processing
Date Created:           19th February 2019
Last Modified:          03rd September 2019
"""
import numpy as np
from scipy import interpolate

class generalUtilities(object):
    def fill_nan(self, data):

        """
        interpolate to fill nan values based on other entries in the row

        :param data: a numpy matrix with nan values
        :return: the matrix with the nan interpolated by the nan values
        """
        ok = ~np.isnan(data)
        xp = ok.ravel().nonzero()[0]
        fp = data[~np.isnan(data)]
        x = np.isnan(data).ravel().nonzero()[0]
        data[np.isnan(data)] = np.interp(x, xp, fp)
        return data

    def convex_hull(self, points):
        """Computes the convex hull of a set of 2D points.

        Input: an iterable sequence of (x, y) pairs representing the points.
        Output: a list of vertices of the convex hull in counter-clockwise order,
          starting from the vertex with the lexicographically smallest coordinates.
        Implements the algorithm CONVEXHULL(P) described by  Mark de Berg, Otfried
        Cheong, Marc van Kreveld, and Mark Overmars, in Computational Geometry:
        Algorithm and Applications, pp. 6-7 in Chapter 1

        :param points: A N X 2 matrix with the wavelengths as the first column
        :return: The convex hull vector
        """
        wvl = np.squeeze(points[:, 0])
        'The starting points be the first two points'
        xcnt, y = points[:2, 0], points[:2, 1]
        'Now iterate over the other points'
        for ii in range(2, points.shape[0], 1):
            'check next prospective convex hull members'
            xcnt = np.append(xcnt, points[ii, 0])
            y = np.append(y, points[ii, 1])
            flag = True

            while (flag == True):
                'Check if a left turn occurs at the central member'
                a1 = (y[-2] - y[-3]) / (xcnt[-2] - xcnt[-3])
                a2 = (y[-1] - y[-2]) / (xcnt[-1] - xcnt[-2])
                if (a2 > a1):
                    xcnt[-2] = xcnt[-1]
                    xcnt = xcnt[:-1]
                    y[-2] = y[-1]
                    y = y[:-1]
                    flag = (xcnt.shape[0] > 2);
                else:
                    flag = False

        f = interpolate.interp1d(xcnt, y)
        ycnt = f(wvl)
        return ycnt

    def colShuffle(self, arr):
        """
        This function will accept a 2D matrix and then shuffles each row
        :param arr: a 2D numpy array
        :return:
        """

        'The shape of the array is'
        x, y = arr.shape

        'The rows are'
        rows = np.indices((x, y))[0]
        'Now shuffle the columns for each row'
        cols = [np.random.permutation(y) for _ in range(x)]
        'Return shuffled matrix'
        return arr[rows, cols]

if __name__ == "__main__":
    obj = generalUtilities()
    print(obj.__class__)




