#!/usr/bin/python3

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from quantimpy import minkowski as mk

def calculate_minkowski(field: np.ndarray, threshold: float = None, geom: bool = False,
                        norm: bool = False) -> tuple:    
    
    """Calculates the Minkowski functional for a 2D input field
    and threshold using the module `quantimpy`, which is found
    here: https://boeleman.github.io/quantimpy/index.html.
    
    geom: if true, the output functionals are normalised as the geometric quantities
        viz., surface area, circumference, and the euler characteristic.
    norm: if true, the output functionals are normalised by the area M0.
    """
    assert field.ndim == 2, "The input field must be 2-dimensional."

    # If no threshold is specified, we use the field itself if it has binary
    # values, otherwise its median value is used as the threshold.
    if threshold is None:
        if type(field[0][0]) is np.bool_:
            binary_field = field
        else:
            threshold = np.percentile(field, 50)
            binary_field = field >= threshold
    else:
        # Define the binary field based on defined threshold
        binary_field = field >= threshold

    M0, M1, M2 = mk.functionals(np.ascontiguousarray(binary_field))

    if geom == True:
        # The functions are normalised such that M0 is the surface area of 
        # the features in the field, M1 is 2pi times the circumference, and
        # M2 is pi times the Euler characteristic.
        M0 = M0
        M1 *= (2*np.pi)
        M2 *= np.pi

    else:
        pass

    if norm == True:
        # The MFs are normalized by the total volume (in 2D the total area)
        M0 /= field.size
        M1 /= field.size
        M2 /= field.size

    else:
        pass
    
    return np.round(M0, 3), np.round(M1, 3), np.round(M2, 3)