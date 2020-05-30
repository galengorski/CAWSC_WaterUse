import numpy as np


def calculate_circle(x, y, radius):
    """
    Method to calculate a polygon of radius r around a point

    Parameters
    ----------
    x : float
        x-coordinate (ie. longitude or easting)
    y : float
        y-coordinate (ie. latitude or Northing)
    radius : float
        radius of circle

    Returns
    -------
        np.array([x,], [y,])
    """
    radians = np.radians(np.arange(0, 361))

    x = x + (np.cos(radians) * radius)
    y = y + (np.sin(radians) * radius)
    return np.array([x, y])
