#!/usr/bin/python
"""script to generate stimuli
"""
import numpy as np
from matplotlib import pyplot as plt
import itertools
import pandas as pd
import scipy.io as sio
import seaborn as sns
import os
import argparse
import json


def bytescale_func(data, cmin=None, cmax=None, high=254, low=0):
    """
    Byte scales an array (image).

    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-254, so that mid is 127).

    If the input image already has dtype uint8, no scaling is done.

    This is copied from scipy.misc, where it is deprecated

    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 254.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.

    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.

    Examples
    --------
    >>> import numpy as np
    >>> from sfp.utils import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def mkR(size, exponent=1, origin=None):
    '''make distance-from-origin (r) matrix

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar)
    containing samples of a radial ramp function, raised to power EXPONENT
    (default = 1), with given ORIGIN (default = (size+1)//2, (0, 0) = upper left).

    NOTE: the origin is not rounded to the nearest int
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xramp, yramp = np.meshgrid(np.arange(1, size[1]+1)-origin[1],
                               np.arange(1, size[0]+1)-origin[0])

    if exponent <= 0:
        # zero to a negative exponent raises:
        # ZeroDivisionError: 0.0 cannot be raised to a negative power
        r = xramp ** 2 + yramp ** 2
        res = np.power(r, exponent / 2.0, where=(r != 0))
    else:
        res = (xramp ** 2 + yramp ** 2) ** (exponent / 2.0)
    return res


def mkAngle(size, phase=0, origin=None):
    '''make polar angle matrix (in radians)

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar)
    containing samples of the polar angle (in radians, CW from the X-axis,
    ranging from -pi to pi), relative to angle PHASE (default = 0), about ORIGIN
    pixel (default = (size+1)/2).

    NOTE: the origin is not rounded to the nearest int
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xramp, yramp = np.meshgrid(np.arange(1, size[1]+1)-origin[1],
                               np.arange(1, size[0]+1)-origin[0])
    xramp = np.array(xramp)
    yramp = np.array(yramp)

    res = np.arctan2(yramp, xramp)

    res = ((res+(np.pi-phase)) % (2*np.pi)) - np.pi

    return res


def log_polar_grating(size, w_r=0, w_a=0, phi=0, ampl=1, origin=None, scale_factor=1):
    """Make a sinusoidal grating in logPolar space.

    this allows for the easy creation of stimuli whose spatial frequency decreases with
    eccentricity, as the peak spatial frequency of neurons in the early visual cortex does.

    Examples
    ============

    radial: `log_polar_grating(512, 4, 10)`

    angular: `log_polar_grating(512, 4, w_a=10)`

    spiral: `log_polar_grating(512, 4, 10, 10)`

    plaid: `log_polar_grating(512, 4, 10) + log_polar_grating(512, 4, w_a=10)`


    Parameters
    =============

    size: scalar. size of the image (only square images permitted).

    w_r: int, logRadial frequency.  Units are matched to those of the angular frequency (`w_a`).

    w_a: int, angular frequency.  Units are cycles per revolution around the origin.

    phi: int, phase (in radians).

    ampl: int, amplitude

    origin: 2-tuple of floats, the origin of the image, from which all distances will be measured
    and angles will be relative to. By default, the center of the image

    scale_factor: int or float. how to scale the distance from the origin before computing the
    grating. this is most often done for checking aliasing; e.g., set size_2 = 100*size_1 and
    scale_factor_2 = 100*scale_factor_1. then the two gratings will have the same pattern, just
    sampled differently
    """
    assert not hasattr(size, '__iter__'), "Only square images permitted, size must be a scalar!"
    rad = mkR(size, origin=origin)/scale_factor
    # if the origin is set such that it lies directly on a pixel, then one of the pixels will have
    # distance 0, that means we'll have a -inf out of np.log2 and thus a nan from the cosine. this
    # little hack avoids that issue.
    if 0 in rad:
        rad += 1e-12
    lrad = np.log2(rad**2)
    theta = mkAngle(size, origin=origin)

    return ampl * np.cos(((w_r * np.log(2))/2) * lrad + w_a * theta + phi)


def _create_better_sampled_grating(orig_size, w_r=0, w_a=0, phi=0, ampl=1, orig_origin=None,
                                   orig_scale_factor=1, check_scale_factor=99):
    if check_scale_factor % 2 == 0:
        raise Exception("For this aliasing check to work, the check_scale_factor must be odd!")
    if orig_origin is None:
        origin = None
    else:
        # this preserves origin's shape, regardless of whether it's an iterable or a scalar
        origin = np.array(orig_origin) * check_scale_factor - (check_scale_factor - 1)/2
    return log_polar_grating(orig_size*check_scale_factor, w_r, w_a, phi, ampl, origin,
                             orig_scale_factor*check_scale_factor)


def aliasing_plot(better_sampled_stim, stim, slices_to_check=None, axes=None, **kwargs):
    """Plot to to check aliasing.

    This does not create the stimuli, only plots them (see `check_aliasing` or `check_aliasing_with
    mask` for functions that create the stimuli and then call this to plot them)

    to add to an existing figure, pass axes (else a new one will be created)
    """
    size = stim.shape[0]
    check_scale_factor = better_sampled_stim.shape[0] // size
    if slices_to_check is None:
        slices_to_check = [(size+1)//2]
    elif not hasattr(slices_to_check, '__iter__'):
        slices_to_check = [slices_to_check]
    if axes is None:
        fig, axes = plt.subplots(ncols=len(slices_to_check), squeeze=False,
                                 figsize=(5*len(slices_to_check), 5), **kwargs)
        # with squeeze=False, this will always be a 2d array, but because we only set ncols, it
        # will only have axes in one dimension
        axes = axes[0]
    x0 = np.array(list(range(size))) / float(size) + 1./(size*2)
    x1 = (np.array(list(range(better_sampled_stim.shape[0]))) / float(better_sampled_stim.shape[0])
          + 1./(better_sampled_stim.shape[0]*2))
    for i, ax in enumerate(axes):
        ax.plot(x1, better_sampled_stim[:, check_scale_factor*slices_to_check[i] +
                                        (check_scale_factor - 1)//2])
        ax.plot(x0, stim[:, slices_to_check[i]], 'o:')


def check_aliasing(size, w_r=0, w_a=0, phi=0, ampl=1, origin=None, scale_factor=1,
                   slices_to_check=None, check_scale_factor=99):
    """Create a simple plot to visualize aliasing

    arguments are mostly the same as for log_polar_grating. this creates both the specified
    stimulus, `orig_stim`, and a `better_sampled_stim`, which has `check_scale_factor` more points
    in each direction. both gratings are returned and a quick plot is generated.

    NOTE that because this requires creating a much larger gradient, it can take a while. Reduce
    `check_scale_factor` to speed it up (at the risk of your "ground truth" becoming aliased)

    slices_to_check: list, None, or int. slices of the stimulus to plot. if None, will plot
    center
    """
    orig_stim = log_polar_grating(size, w_r, w_a, phi, ampl, origin, scale_factor)
    better_sampled_stim = _create_better_sampled_grating(size, w_r, w_a, phi, ampl, origin,
                                                         scale_factor, check_scale_factor)
    aliasing_plot(better_sampled_stim, orig_stim, slices_to_check)
    return orig_stim, better_sampled_stim


def _fade_mask(mask, inner_number_of_fade_pixels, outer_number_of_fade_pixels, origin=None):
    """note that mask must contain 0s where you want to mask out, 1s elsewhere
    """
    # if there's no False in mask, then we don't need to mask anything out. and if there's only
    # False, we don't need to fade anything. and if there's no fade pixels, then we don't fade
    # anything
    if False not in mask or True not in mask or (inner_number_of_fade_pixels == 0 and
                                                 outer_number_of_fade_pixels == 0):
        return mask
    size = mask.shape[0]
    rad = mkR(size, origin=origin)
    inner_rad = (mask*rad)[(mask*rad).nonzero()].min()
    # in this case, there really isn't an inner radius, just an outer one, so we ignore this
    if inner_rad == rad.min():
        inner_rad = 0
        inner_number_of_fade_pixels = 0
    outer_rad = (mask*rad).max()

    # in order to get the right number of pixels to act as transition, we set the frequency based
    # on the specified number_of_fade_pixels
    def inner_fade(x):
        if inner_number_of_fade_pixels == 0:
            return (-np.cos(2*np.pi*(x-inner_rad) / (size/2.))+1)/2
        inner_fade_freq = (size/2.) / (2*inner_number_of_fade_pixels)
        return (-np.cos(inner_fade_freq*2*np.pi*(x-inner_rad) / (size/2.))+1)/2

    def outer_fade(x):
        if outer_number_of_fade_pixels == 0:
            return (-np.cos(2*np.pi*(x-outer_rad) / (size/2.))+1)/2
        outer_fade_freq = (size/2.) / (2*outer_number_of_fade_pixels)
        return (-np.cos(outer_fade_freq*2*np.pi*(x-outer_rad) / (size/2.))+1)/2

    faded_mask = np.piecewise(rad,
                              [rad < inner_rad,
                               (rad >= inner_rad) & (rad <= (inner_rad + inner_number_of_fade_pixels)),
                               (rad > (inner_rad + inner_number_of_fade_pixels)) & (rad < outer_rad - outer_number_of_fade_pixels),
                               (rad >= outer_rad - outer_number_of_fade_pixels) & (rad <= (outer_rad)),
                               (rad > (outer_rad))],
                              [0, inner_fade, 1, outer_fade, 0])
    return faded_mask


def _calc_sf_analytically(x, y, stim_type='logpolar', w_r=None, w_a=None, w_x=None, w_y=None):
    """helper function that calculates spatial frequency (in cpp)

    this should NOT be called directly. it is the function that gets called by `sf_cpp` and
    `create_sf_maps_cpp`.
    """
    if stim_type == 'logpolar':
        if w_r is None or w_a is None or w_x is not None or w_y is not None:
            raise Exception("When stim_type is %s, w_r / w_a must be set and w_x / w_y must be"
                            " None!" % stim_type)
    elif stim_type == 'constant':
        if w_r is not None or w_a is not None or w_x is None or w_y is None:
            raise Exception("When stim_type is constant, w_x / w_y must be set and w_a / w_r must"
                            " be None!")
    else:
        raise Exception("Don't know how to handle stim_type %s!" % stim_type)
    # we want to approximate the spatial frequency of our log polar gratings. We can do that using
    # the first two terms of the Taylor series. Since our gratings are of the form cos(g(X)) (where
    # X contains both x and y values), then to approximate them at location X_0, we'll use
    # cos(g(X_0) + g'(X_0)(X-X_0)), where g'(X_0) is the derivative of g at X_0 (with separate x
    # and y components). g(X_0) is the phase of the approximation and so not important here, but
    # that g'(X_0) is the local spatial frequency that we're interested in. Thus we take the
    # derivative of our log polar grating function with respect to x and y in order to get dx and
    # dy, respectively (after some re-arranging and cleaning up). the constant stimuli, by
    # definition, have a constant spatial frequency every where in the image.
    if stim_type == 'logpolar':
        dy = (y * w_r + w_a * x) / (x**2 + y**2)
        dx = (x * w_r - w_a * y) / (x**2 + y**2)
    elif stim_type == 'constant':
        try:
            size = x.shape
            dy = w_y * np.ones((size, size))
            dx = w_x * np.ones((size, size))
        # if x is an int, this will raise a SyntaxError; if it's a float, it will raise an
        # AttributeError; if it's an array with a single value (e.g., np.array(1), not
        # np.array([1])), then it will raise a TypeError
        except (SyntaxError, TypeError, AttributeError):
            dy = w_y
            dx = w_x
    if stim_type == 'logpolar':
        # Since x, y are in pixels (and so run from ~0 to ~size/2), dx and dy need to be divided by
        # 2*pi in order to get the frequency in cycles / pixel. This is analogous to the 1d case:
        # if x runs from 0 to 1 and f(x) = cos(w * x), then the number of cycles in f(x) is w /
        # 2*pi. (the values for the constant stimuli are given in cycles per pixel already)
        dy /= 2*np.pi
        dx /= 2*np.pi
    # I want this to lie between 0 and 2*pi, because otherwise it's confusing
    direction = np.mod(np.arctan2(dy, dx), 2*np.pi)
    return dx, dy, np.sqrt(dx**2 + dy**2), direction


def sf_cpp(eccen, angle, stim_type='logpolar', w_r=None, w_a=None, w_x=None, w_y=None):
    """calculate the spatial frequency in cycles per pixel.

    this function returns spatial frequency values; it returns values that give the spatial
    frequency at the point specified by x, y (if you instead want a map showing the spatial
    frequency everywhere in the specified stimulus, use `create_sf_maps_cpp`). returns four values:
    the spatial frequency in the x direction (dx), the spatial frequency in the y direction (dy),
    the magnitude (sqrt(dx**2 + dy**2)) and the direction (arctan2(dy, dx))

    In most cases, you want the magnitude, as this is the local spatial frequency of the specified
    grating at that point.

    NOTE: for this to work, the zero for the angle you're passing in must correspond to the right
    horizontal meridian, angle should lie between 0 and 2*pi, and you should move clockwise as
    angle increases. This is all so it corresponds to the values for the direction of the spatial
    frequency.

    eccen, angle: floats. The location you want to find the spatial frequency for, in polar
    coordinates. eccen should be in pixels, NOT degrees. angle should be in radians.

    stim_type: {'logpolar', 'constant'}. which type of stimuli to generate the spatial frequency
    map for. This matters because we determine the spatial frequency maps analytically and so
    *cannot* do so in a stimulus-driven manner. if 'logpolar', the log-polar gratings created by
    log_polar_grating. if 'constant', the constant gratings created by create_sin_cpp (and
    gen_constant_stim_set). If 'constant', then w_x and w_y must be set, w_r and w_a must be None;
    if 'logpolar', then the opposite.

    """
    x = eccen * np.cos(angle)
    y = eccen * np.sin(angle)
    if x == 0:
        x += 1e-12
    if y == 0:
        y += 1e-12
    return _calc_sf_analytically(x, y, stim_type, w_r, w_a, w_x, w_y)


def sf_cpd(eccen, angle, pixel_diameter=714, degree_diameter=8.4, stim_type='logpolar', w_r=None,
           w_a=None, w_x=None, w_y=None):
    """calculate the spatial frequency in cycles per degree.

    this function returns spatial frequency values; it returns values that give the spatial
    frequency at the point specified by x, y (if you instead want a map showing the spatial
    frequency everywhere in the specified stimulus, use `create_sf_maps_cpp`). returns four values:
    the spatial frequency in the x direction (dx), the spatial frequency in the y direction (dy),
    the magnitude (sqrt(dx**2 + dy**2)) and the direction (arctan2(dy, dx))

    In most cases, you want the magnitude, as this is the local spatial frequency of the specified
    grating at that point.

    NOTE: for this to work, the zero for the angle you're passing in must correspond to the right
    horizontal meridian, angle should lie between 0 and 2*pi, and you should move clockwise as
    angle increases. This is all so it corresponds to the values for the direction of the spatial
    frequency.

    degree_diameter: int, the visual angle (in degrees) corresponding to the diameter of the full
    image

    eccen, angle: floats. The location you want to find the spatial frequency for, in polar
    coordinates. eccen should be in degrees (NOT pixels). angle should be in radians.

    stim_type: {'logpolar', 'constant'}. which type of stimuli to generate the spatial frequency
    map for. This matters because we determine the spatial frequency maps analytically and so
    *cannot* do so in a stimulus-driven manner. if 'logpolar', the log-polar gratings created by
    log_polar_grating. if 'constant', the constant gratings created by create_sin_cpp (and
    gen_constant_stim_set). If 'constant', then w_x and w_y must be set, w_r and w_a must be None;
    if 'logpolar', the opposite.

    """
    conversion_factor = degree_diameter / pixel_diameter
    # this is in degrees, so we divide it by deg/pix to get the eccen in pix
    eccen /= conversion_factor
    dx, dy, magnitude, direction = sf_cpp(eccen, angle, stim_type, w_r, w_a, w_x, w_y)
    # these are all in cyc/pix, so we divide them by deg/pix to get them in cyc/deg
    dx /= conversion_factor
    dy /= conversion_factor
    magnitude /= conversion_factor
    return dx, dy, magnitude, direction


def sf_origin_polar_cpd(eccen, angle, pixel_diameter=714, degree_diameter=8.4,
                        stim_type='logpolar', w_r=None, w_a=None, w_x=None, w_y=None):
    """calculate the local origin-referenced polar spatial frequency (radial/angular) in cpd

    returns the local spatial frequency with respect to the radial and angular directions.

    NOTE: for this to work, the zero for the angle you're passing in must correspond to the right
    horizontal meridian, angle should lie between 0 and 2*pi, and you should move clockwise as
    angle increases. This is all so it corresponds to the values for the direction of the spatial
    frequency.

    pixel_diameter: int, the visual angle (in degrees) corresponding to the diameter of the full
    image

    eccen, angle: floats. The location you want to find the spatial frequency for, in polar
    coordinates. eccen should be in degrees (NOT pixels). angle should be in radians.

    stim_type: {'logpolar', 'constant'}. which type of stimuli to generate the spatial frequency
    map for. This matters because we determine the spatial frequency maps analytically and so
    *cannot* do so in a stimulus-driven manner. if 'logpolar', the log-polar gratings created by
    log_polar_grating. if 'constant', the constant gratings created by create_sin_cpp (and
    gen_constant_stim_set). If 'constant', then w_x and w_y must be set, w_r and w_a must be None;
    if 'logpolar', then the opposite.

    """
    _, _, mag, direc = sf_cpd(eccen, angle, pixel_diameter, degree_diameter, stim_type, w_r, w_a,
                              w_x, w_y)
    new_angle = np.mod(direc - angle, 2*np.pi)
    dr = mag * np.cos(new_angle)
    da = mag * np.sin(new_angle)
    return dr, da, new_angle


def create_sf_maps_cpp(pixel_diameter=714, origin=None, scale_factor=1, stim_type='logpolar',
                       w_r=None, w_a=None, w_x=None, w_y=None):
    """Create maps of spatial frequency in cycles per pixel.

    this function creates spatial frequency maps; that is, it returns images that show the spatial
    frequency everywhere in the specified stimulus (if you instead want the spatial frequency at a
    specific point, use `sf_cpp`). returns four maps: the spatial frequency in the x direction
    (dx), the spatial frequency in the y direction (dy), the magnitude (sqrt(dx**2 + dy**2)) and
    the direction (arctan2(dy, dx))

    In most cases, you want the magnitude, as this is the local spatial frequency of the
    corresponding log polar grating at that point.

    stim_type: {'logpolar', 'constant'}. which type of stimuli to generate the spatial frequency
    map for. This matters because we determine the spatial frequency maps analytically and so
    *cannot* do so in a stimulus-driven manner. if 'logpolar', the log-polar gratings created by
    log_polar_grating. if 'constant', the constant gratings created by create_sin_cpp (and
    gen_constant_stim_set). If 'constant', then w_x and w_y must be set, w_r and w_a must be None;
    if 'logpolar', then the opposite.

    """
    assert not hasattr(pixel_diameter, '__iter__'), "Only square images permitted, pixel_diameter must be a scalar!"
    pixel_diameter = int(pixel_diameter)
    if origin is None:
        origin = ((pixel_diameter+1) / 2., (pixel_diameter+1) / 2.)
    # we do this in terms of x and y
    x, y = np.divide(np.meshgrid(np.array(list(range(1, pixel_diameter+1))) - origin[0],
                                 np.array(list(range(1, pixel_diameter+1))) - origin[1]),
                     scale_factor)
    # if the origin is set such that it lies directly on a pixel, then one of the pixels will have
    # distance 0 and that means we'll have a divide by zero coming up. this little hack avoids that
    # issue.
    if 0 in x:
        x += 1e-12
    if 0 in y:
        y += 1e-12
    return _calc_sf_analytically(x, y, stim_type, w_r, w_a, w_x, w_y)


def create_sf_maps_cpd(pixel_diameter=714, degree_diameter=7.4, origin=None, scale_factor=1,
                       stim_type='logpolar', w_r=None, w_a=None, w_x=None, w_y=None):
    """Create map of the spatial frequency in cycles per degree of visual angle

    this function creates spatial frequency maps; that is, it returns images that show the spatial
    frequency everywhere in the specified stimulus (if you instead want the spatial frequency at a
    specific point, use `sf_cpp`). returns four maps: the spatial frequency in the x direction
    (dx), the spatial frequency in the y direction (dy), the magnitude (sqrt(dx**2 + dy**2)) and
    the direction (arctan2(dy, dx))

    In most cases, you want the magnitude, as this is the local spatial frequency of the
    corresponding log polar grating at that point

    """
    conversion_factor = degree_diameter / pixel_diameter
    dx, dy, mag, direc = create_sf_maps_cpp(pixel_diameter, origin, scale_factor, stim_type, w_r,
                                            w_a, w_x, w_y)
    dx /= conversion_factor
    dy /= conversion_factor
    mag /= conversion_factor
    return dx, dy, mag, direc


def create_sf_origin_polar_maps_cpd(pixel_diameter=714, degree_diameter=8.4, origin=None,
                                    scale_factor=1, stim_type='logpolar', w_r=None, w_a=None,
                                    w_x=None, w_y=None):
    """create map of the origin-referenced polar spatial frequency (radial/angular) in cpd

    returns maps of the spatial frequency with respect to the radial and angular directions.

    degree_diameter: int, the visual angle (in degrees) corresponding to the diameter of the full
    image

    stim_type: {'logpolar', 'constant'}. which type of stimuli to generate the spatial frequency
    map for. This matters because we determine the spatial frequency maps analytically and so
    *cannot* do so in a stimulus-driven manner. if 'logpolar', the log-polar gratings created by
    log_polar_grating. if 'constant', the constant gratings created by create_sin_cpp (and
    gen_constant_stim_set).If 'constant', then w_x and w_y must be set, w_r and w_a must be None;
    if 'logpolar', then the opposite.

    """
    _, _, mag, direc = create_sf_maps_cpd(pixel_diameter, degree_diameter, origin, scale_factor,
                                          stim_type, w_r, w_a, w_x, w_y)
    angle = mkAngle(pixel_diameter, origin=origin)
    new_angle = np.mod(direc - angle, 2*np.pi)
    dr = mag * np.cos(new_angle)
    da = mag * np.sin(new_angle)
    return dr, da, new_angle


def create_antialiasing_mask(size, w_r=0, w_a=0, origin=None, number_of_fade_pixels=3,
                             scale_factor=1):
    """Create mask to hide aliasing

    Because of how our stimuli are created, they have higher spatial frequency at the origin
    (probably center of the image) than at the edge of the image. This makes it a little harder to
    determine where aliasing will happen. for the specified arguments, this will create the mask
    that will hide the aliasing of the grating(s) with these arguments.

    the mask will not be strictly binary, there will a `number_of_fade_pixels` where it transitions
    from 0 to 1. this transition is half of a cosine.

    returns both the faded_mask and the binary mask.
    """
    _, _, mag, _ = create_sf_maps_cpp(size, origin, scale_factor, w_r=w_r, w_a=w_a)
    # the nyquist frequency is .5 cycle per pixel, but we make it a lower to give ourselves a
    # little fudge factor
    nyq_freq = .475
    mask = mag < nyq_freq
    faded_mask = _fade_mask(mask, number_of_fade_pixels, 0, origin)
    return faded_mask, mask


def create_outer_mask(size, origin, radius=None, number_of_fade_pixels=3):
    """Create mask around the outside of the image

    this gets us a window that creates a circular (or some subset of circular) edge. this returns
    both the faded and the unfaded versions.

    radius: float or None. the radius, in pixels, of the mask. Everything farther away from the
    origin than this will be masked out. If None, we pick radius such that it's the distance to the
    edge of the square image. If horizontal and vertical have different distances, we will take the
    shorter of the two. If the distance from the origin to the horizontal edge is not identical in
    both directions, we'll take the longer of the two (similar for vertical).

    To combine this with the antialiasing mask, call np.logical_and on the two unfaded masks (and
    then fade that if you want to fade it)
    """
    rad = mkR(size, origin=origin)
    assert not hasattr(size, "__iter__"), "size must be a scalar!"
    if radius is None:
        radius = min(rad[:, size//2].max(), rad[size//2, :].max())
    mask = rad < radius
    return _fade_mask(mask, 0, number_of_fade_pixels, origin), mask


def check_aliasing_with_mask(size, w_r=0, w_a=0, phi=0, ampl=1, origin=None, scale_factor=1,
                             number_of_fade_pixels=3, slices_to_check=None, check_scale_factor=99):
    """check the aliasing when mask is applied
    """
    stim = log_polar_grating(size, w_r, w_a, phi, ampl, origin, scale_factor)
    fmask, mask = create_antialiasing_mask(size, w_r, w_a, origin)
    better_sampled_stim = _create_better_sampled_grating(size, w_r, w_a, phi, ampl, origin,
                                                         scale_factor, check_scale_factor)
    big_fmask = fmask.repeat(check_scale_factor, 0).repeat(check_scale_factor, 1)
    big_mask = mask.repeat(check_scale_factor, 0).repeat(check_scale_factor, 1)
    if slices_to_check is None:
        slices_to_check = [(size+1)//2]
    fig, axes = plt.subplots(ncols=3, nrows=len(slices_to_check), squeeze=False,
                             figsize=(15, 5*len(slices_to_check)))
    aliasing_plot(better_sampled_stim, stim, slices_to_check, axes[:, 0])
    aliasing_plot(big_fmask*better_sampled_stim, fmask*stim, slices_to_check, axes[:, 1])
    aliasing_plot(big_mask*better_sampled_stim, mask*stim, slices_to_check, axes[:, 2])
    axes[0, 0].set_title("Slices of un-masked stimulus")
    axes[0, 1].set_title("Slices of fade-masked stimulus")
    axes[0, 2].set_title("Slices of binary-masked stimulus")
    return stim, fmask, mask, better_sampled_stim, big_fmask, big_mask


def find_ecc_range_in_pixels(stim, mid_val=127):
    """find the min and max eccentricity of the stimulus, in pixels

    all of our stimuli have a central aperture where nothing is presented and an outside limit,
    beyond which nothing is presented.

    this assumes the fixation is in the center of the stimulus, will have to re-think things if
    it's not.

    returns min, max
    """
    if stim.ndim == 3:
        stim = stim[0, :, :]
    R = mkR(stim.shape)
    x, y = np.where(stim != mid_val)
    return R[x, y].min(), R[x, y].max()


def find_ecc_range_in_degrees(stim, degree_radius, mid_val=127):
    """find the min and max eccentricity of the stimulus, in degrees

    all of our stimuli have a central aperture where nothing is presented and an outside limit,
    beyond which nothing is presented. In order to make sure we're not looking at voxels whose pRFs
    lie outside the stimulus, we want to know the extent of the stimulus annulus, in degrees

    this assumes the fixation is in the center of the stimulus, will have to re-think things if
    it's not.

    stim_rad_deg: int or float, the radius of the stimulus, in degrees.

    returns min, max
    """
    if stim.ndim == 3:
        stim = stim[0, :, :]
    Rmin, Rmax = find_ecc_range_in_pixels(stim, mid_val)
    R = mkR(stim.shape)
    # if stim_rad_deg corresponds to the max vertical/horizontal extent, the actual max will be
    # np.sqrt(2*stim_rad_deg**2) (this corresponds to the far corner). this should be the radius of
    # the screen, because R starts from the center and goes to the edge
    factor = R.max() / np.sqrt(2*degree_radius**2)
    return Rmin / factor, Rmax / factor


def calculate_stim_local_sf(stim, w_1, w_2, stim_type, eccens, angles, degree_radius=4.2,
                            plot_flag=False, mid_val=127):
    """calculate the local spatial frequency for a specified stimulus and screen size

    stim: 2d array of floats. an example stimulus. used to determine where the stimuli are masked
    (and thus where the spatial frequency is zero).

    w_1, w_2: ints or floats. the first and second components of the stimulus's spatial
    frequency. if stim_type is 'logarpolar', this should be the radial and angular components (in
    that order!); if stim_type is 'constant', this should be the x and y components (in that
    order!)

    stim_type: {'logpolar', 'constant'}. which type of stimuli were used in the session we're
    analyzing. This matters because it changes the local spatial frequency and, since that is
    determined analytically and not directly from the stimuli, we have no way of telling otherwise.

    eccens, angles: lists of floats. these are the eccentricities and angles we want to find
    local spatial frequency for.

    degree_radius: float, the radius of the stimulus, in degrees of visual angle

    plot_flag: boolean, optional, default False. Whether to create a plot showing the local spatial
    frequency vs eccentricity for the specified stimulus

    mid_val: int. the value of mid-grey in the stimuli, should be 127 or 128

    """
    eccen_min, eccen_max = find_ecc_range_in_degrees(stim, degree_radius, mid_val)
    eccen_local_freqs = []
    for i, (e, a) in enumerate(zip(eccens, angles)):
        if stim_type == 'logpolar':
            dx, dy, mag, direc = sf_cpd(e, a, stim.shape[0], degree_radius*2,
                                        stim_type=stim_type, w_r=w_1, w_a=w_2)
            dr, da, new_angle = sf_origin_polar_cpd(e, a, stim.shape[0], degree_radius*2,
                                                    stim_type=stim_type, w_r=w_1, w_a=w_2)
        elif stim_type == 'constant':
            dx, dy, mag, direc = sf_cpd(e, a, stim.shape[0], degree_radius*2, stim_type=stim_type,
                                        w_x=w_1, w_y=w_2)
            dr, da, new_angle = sf_origin_polar_cpd(e, a, stim.shape[0], degree_radius*2,
                                                    stim_type=stim_type, w_x=w_1, w_y=w_2)
        eccen_local_freqs.append(pd.DataFrame(
            {'local_w_x': dx, 'local_w_y': dy, 'local_w_r': dr, 'local_w_a': da, 'eccen': e,
             'angle': a, 'local_sf_magnitude': mag, 'local_sf_xy_direction': direc,
             'local_sf_ra_direction': new_angle}, [i]))
    eccen_local_freqs = pd.concat(eccen_local_freqs)

    if plot_flag:
        plt.plot(eccen_local_freqs['eccen'], eccen_local_freqs['local_sf_magnitude'])
        ax = plt.gca()
        ax.set_title('Spatial frequency vs eccentricity')
        ax.set_xlabel('Eccentricity (degrees)')
        ax.set_ylabel('Local spatial frequency (cpd)')

    return eccen_local_freqs


def check_stim_properties(pixel_diameter=714, origin=None, degree_diameter=8.4, w_r=0,
                          w_a=range(10), eccen_range=(1, 4.2)):
    """Creates a dataframe with data on several stimulus properties, based on the specified arguments

    the properties examined are:
    - mask radius in pixels
    - mask radius in degrees
    - max frequency in cycles per pixel
    - min frequency in cycles per pixel
    - max frequency in cycles per degree
    - min frequency in cycles per degree
    - max masked frequency in cycles per pixel
    - max masked frequency in cycles per degree

    we also return a second dataframe, sf_df, which contains the local spatial frequency of each
    (unmasked) stimulus at each eccentricity, in cycles per pixel and cycles per degree. we only
    examine the eccentricities within eccen_range, and we bin by degree, averaging within each
    bin. that is, with eccen_range=(1, 5), we calculate the average local spatial frequency of a
    given stimulus from 1 to 2 degrees, 2 to 3 degrees, ..., 4 to 5 degrees.

    Note that we don't calculate the min masked frequency because that will always be zero (because
    we zero out the center of the image, where the frequency is at its highest).

    note that pixel_diameter, origin, and degree_diameter must have only one value, w_r and w_a can
    be lists or single values (and all combinations of them will be checked)

    """
    if hasattr(pixel_diameter, '__iter__'):
        raise Exception("pixel_diameter must *not* be iterable! All generated stimuli must be the same pixel_diameter")
    if hasattr(origin, '__iter__'):
        raise Exception("only one value of origin at a time!")
    if hasattr(degree_diameter, '__iter__'):
        raise Exception("only one value of degree_diameter at a time!")
    if not hasattr(w_r, '__iter__'):
        w_r = [w_r]
    if not hasattr(w_a, '__iter__'):
        w_a = [w_a]
    rad = mkR(pixel_diameter, origin=origin)
    mask_df = []
    sf_df = []
    eccens = [(i+i+1)/2 for i in np.linspace(*eccen_range, 10)]
    angles = [0 for i in eccens]
    for i, (f_r, f_a) in enumerate(itertools.product(w_r, w_a)):
        fmask, mask = create_antialiasing_mask(pixel_diameter, f_r, f_a, origin, 0)
        _, _, mag_cpp, _ = create_sf_maps_cpp(pixel_diameter, origin, w_r=f_r, w_a=f_a)
        _, _, mag_cpd, _ = create_sf_maps_cpd(pixel_diameter, degree_diameter, origin, w_r=f_r, w_a=f_a)
        data = {'mask_radius_pix': (~mask*rad).max(), 'w_r': f_r, 'w_a': f_a,
                'freq_distance': np.sqrt(f_r**2 + f_a**2)}
        data['mask_radius_deg'] = data['mask_radius_pix'] / (rad.max() / np.sqrt(2*(degree_diameter/2.)**2))
        for name, mag in zip(['cpp', 'cpd'], [mag_cpp, mag_cpd]):
            data[name + "_max"] = mag.max()
            data[name + "_min"] = mag.min()
            data[name + "_masked_max"] = (fmask * mag).max()
        mask_df.append(pd.DataFrame(data, index=[i]))
        sf = calculate_stim_local_sf(np.ones((pixel_diameter, pixel_diameter)), f_r, f_a,
                                                          'logpolar', eccens, angles,
                                                          degree_diameter/2)
        sf = sf.rename(columns={'local_sf_magnitude': 'local_freq_cpd'})
        sf['w_r'] = f_r
        sf['w_a'] = f_a
        sf['local_freq_cpp'] = sf['local_freq_cpd'] / (rad.max() / np.sqrt(2*(degree_diameter/2.)**2))
        # period is easier to think about
        sf['local_period_ppc'] = 1. / sf['local_freq_cpp']
        sf['local_period_dpc'] = 1. / sf['local_freq_cpd']
        sf_df.append(sf.reset_index())
    return pd.concat(mask_df), pd.concat(sf_df).reset_index(drop=True)


def _set_ticklabels(datashape):
    xticklabels = datashape[1]//10
    if xticklabels == 0 or xticklabels == 1:
        xticklabels = True
    yticklabels = datashape[0]//10
    if yticklabels == 0 or yticklabels == 1:
        yticklabels = True
    return xticklabels, yticklabels


def plot_stim_properties(mask_df, x='w_a', y='w_r', data_label='mask_radius_pix',
                         title_text="Mask radius in pixels",
                         fancy_labels={"w_a": r"$\omega_a$", "w_r": r"$\omega_r$"},
                         **kwargs):
    """plot the mask_df created by check_mask_radius, to visualize how mask radius depends on args.

    fancy_labels is a dict of mask_df columns to nice (latex) ways of labeling them on the plot.
    """
    def facet_heatmap(x, y, data_label, **kwargs):
        data = kwargs.pop('data').pivot(y, x, data_label)
        xticks, yticks = _set_ticklabels(data.shape)
        sns.heatmap(data, xticklabels=xticks, yticklabels=yticks, **kwargs).invert_yaxis()

    cmap = kwargs.pop('cmap', 'Blues')
    font_scale = kwargs.pop('font_scale', 1.5)
    plotting_context = kwargs.pop('plotting_context', 'notebook')
    size = kwargs.pop('size', 3)
    with sns.plotting_context(plotting_context, font_scale=font_scale):
        g = sns.FacetGrid(mask_df, size=size)
        cbar_ax = g.fig.add_axes([.92, .3, .02, .4])
        g.map_dataframe(facet_heatmap, x, y, data_label, vmin=0,
                        vmax=mask_df[data_label].max(), cmap=cmap, cbar_ax=cbar_ax, **kwargs)
        g.fig.suptitle(title_text)
        g.fig.tight_layout(rect=[0, 0, .9, .95])
        g.set_axis_labels(fancy_labels[x], fancy_labels[y])


def gen_log_polar_stim_set(size, freqs_ra=[(0, 0)], phi=[0], ampl=[1], origin=None,
                           number_of_fade_pixels=3, combo_stimuli_type=['spiral'], bytescale=True):
    """Generate the specified set of log-polar stimuli and apply the anti-aliasing mask

    this function creates the specified log-polar stimuli, calculates what their anti-aliasing
    masks should be, and applies the largest of those masks to all stimuli. It also applies an
    outer mask so each of them is surrounded by faded, circular mask.

    Note that this function should be run *last*, after you've determined your parameters and
    checked to make sure the aliasing is taken care of.

    Parameters
    =============

    freqs_ra: list of tuples of floats. the frequencies (radial and angular, in that order) of the
    stimuli to create. Each entry in the list corresponds to one stimuli, which will use the
    specified (w_r, w_a).

    combo_stimuli_type: list with possible elements {'spiral', 'plaid'}. type of stimuli to create
    when both w_r and w_a are nonzero, as described in the docstring for log_polar_grating (to
    create radial and angular stimuli, just include 0 in w_a or w_r, respectively).

    bytescale: boolean, default True. if True, calls bytescale(cmin=-1, cmax=1) on image to rescale
    it to between 0 and 254 (mid-value is 127), with dtype uint8. this is done because this is
    probably sufficient for displays and takes up much less space.


    Returns
    =============

    masked stimuli, unmasked stimuli, and the mask used to mask the stimuli

    """
    # we need to make sure that size, origin, and number_of_fade_pixels are not iterable and the
    # other arguments are
    if hasattr(size, '__iter__'):
        raise Exception("size must *not* be iterable! All generated stimuli must be the same size")
    if hasattr(origin, '__iter__'):
        raise Exception("origin must *not* be iterable! All generated stimuli must have the same "
                        " origin")
    if hasattr(number_of_fade_pixels, '__iter__'):
        raise Exception("number_of_fade_pixels must *not* be iterable! It's a property of the mask"
                        " and we want to apply the same mask to all stimuli.")
    # this isn't a typo: we want to make sure that freqs_ra is a list of tuples; an easy way to
    # check is to make sure the *entries* of freqs_ra are iterable
    if not hasattr(freqs_ra[0], '__iter__'):
        freqs_ra = [freqs_ra]
    if not hasattr(phi, '__iter__'):
        phi = [phi]
    if not hasattr(ampl, '__iter__'):
        ampl = [ampl]
    if not hasattr(combo_stimuli_type, '__iter__'):
        combo_stimuli_type = [combo_stimuli_type]
    stimuli = []
    masked_stimuli = []
    mask = []
    for w_r, w_a in freqs_ra:
        _, tmp_mask = create_antialiasing_mask(size, w_r, w_a, origin, number_of_fade_pixels)
        mask.append(tmp_mask)
    mask.append(create_outer_mask(size, origin, None, number_of_fade_pixels)[1])
    if len(mask) > 1:
        mask = np.logical_and.reduce(mask)
    else:
        mask = mask[0]
    mask = _fade_mask(mask, number_of_fade_pixels, number_of_fade_pixels, origin)
    for (w_r, w_a), p, A in itertools.product(freqs_ra, phi, ampl):
        if w_r == 0 and w_a == 0:
            # this is the empty stimulus
            continue
        if 0 in [w_r, w_a] or 'spiral' in combo_stimuli_type:
            tmp_stimuli = log_polar_grating(size, w_r, w_a, p, A, origin)
            if bytescale:
                masked_stimuli.append(bytescale_func(tmp_stimuli*mask, cmin=-1, cmax=1))
                stimuli.append(bytescale_func(tmp_stimuli, cmin=-1, cmax=1))
            else:
                masked_stimuli.append(tmp_stimuli*mask)
                stimuli.append(tmp_stimuli)
        if 'plaid' in combo_stimuli_type and 0 not in [w_r, w_a]:
            tmp_stimuli = (log_polar_grating(size, w_r, 0, p, A, origin) +
                           log_polar_grating(size, 0, w_a, p, A, origin))
            if bytescale:
                masked_stimuli.append(bytescale_func(tmp_stimuli*mask, cmin=-1, cmax=1))
                stimuli.append(bytescale_func(tmp_stimuli, cmin=-1, cmax=1))
            else:
                masked_stimuli.append(tmp_stimuli*mask)
                stimuli.append(tmp_stimuli)
    return masked_stimuli, stimuli, mask


def create_sin_cpp(size, w_x, w_y, phase=0, origin=None):
    """create a full 2d sine wave, with frequency in cycles / pixel
    """
    if origin is None:
        origin = [(size+1) / 2., (size+1) / 2.]
    x = np.array(range(1, size+1))
    x, y = np.meshgrid(x - origin[0], x - origin[1])
    return np.cos(2*np.pi*x*w_x + 2*np.pi*y*w_y + phase)


def gen_constant_stim_set(size, mask, freqs_xy=[(0, 0)], phi=[0], ampl=[1], origin=None,
                          bytescale=True):
    """Generate the specified set of constant grating stimuli and apply the supplied mask

    this function creates the specified constant grating stimuli and applies the supplied mask to
    all stimuli. It also applies an outer mask so each of them is surrounded by faded, circular
    mask.

    Note that this function should be run *last*, after you've determined your parameters and
    checked to make sure the aliasing is taken care of.

    Parameters
    =============

    freqs_xy: list of tuples of floats. the frequencies (x and y, in that order) of the stimuli to
    create. Each entry in the list corresponds to one stimuli, which will use the specified (w_x,
    w_y). They sould be in cycles per pixel.

    bytescale: boolean, default True. if True, calls bytescale(cmin=-1, cmax=1) on image to rescale
    it to between 0 and 254 (mid-value is 127), with dtype uint8. this is done because this is
    probably sufficient for displays and takes up much less space.


    Returns
    =============

    masked stimuli and unmasked stimuli

    """
    # we need to make sure that size, origin, and number_of_fade_pixels are not iterable and the
    # other arguments are
    if hasattr(size, '__iter__'):
        raise Exception("size must *not* be iterable! All generated stimuli must be the same size")
    if hasattr(origin, '__iter__'):
        raise Exception("origin must *not* be iterable! All generated stimuli must have the same "
                        " origin")
    # this isn't a typo: we want to make sure that freqs_xy is a list of tuples; an easy way to
    # check is to make sure the *entries* of freqs_xy are iterable
    if not hasattr(freqs_xy[0], '__iter__'):
        freqs_xy = [freqs_xy]
    if not hasattr(phi, '__iter__'):
        phi = [phi]
    if not hasattr(ampl, '__iter__'):
        ampl = [ampl]
    stimuli = []
    masked_stimuli = []
    for (w_x, w_y), p, A in itertools.product(freqs_xy, phi, ampl):
        if w_x == 0 and w_y == 0:
            # this is the empty stimulus
            continue
        else:
            tmp_stimuli = A * create_sin_cpp(size, w_x, w_y, p, origin=origin)
            if bytescale:
                masked_stimuli.append(bytescale_func(tmp_stimuli*mask, cmin=-1, cmax=1))
                stimuli.append(bytescale_func(tmp_stimuli, cmin=-1, cmax=1))
            else:
                masked_stimuli.append(tmp_stimuli*mask)
                stimuli.append(tmp_stimuli)
    return masked_stimuli, stimuli


def _gen_freqs(base_freqs, n_orientations=4, n_intermed_samples=2, round_flag=True):
    """turn the base frequencies into the full set.

    base frequencies are the distance from the center of frequency space.

    n_orientations: int, the number of "canonical orientations". That is, the number of
    orientations that should use the base_freqs. We will equally sample orientation space, so that
    if n_orientations==4, then we'll use angles 0, pi/4, pi/2, 3*pi/4

    n_intermed_samples: int, the number of samples at the intermediate frequency. In order to
    sample some more orientations, we pick the middle frequency out of base_freqs and then sample
    n_intermed_samples times between the canonical orientations at that frequency. For example, if
    this is 2 and n_orientations is 4, we will sample the intermediate frequency at pi/12, 2*pi/12,
    4*pi/12, 5*pi/12, 7*pi/12, 8*pi/12, 10*pi/12, 11*pi/12.

    """
    intermed_freq = base_freqs[len(base_freqs)//2]
    ori_angles = [np.pi*1/n_orientations*i for i in range(n_orientations)]
    # the following determines how to step through the angles so to get n_intermed_angles different
    # intermediary angles
    n_intermed_steps = int(n_orientations * (n_intermed_samples+1))
    intermed_locs = [i for i in range(n_intermed_steps)
                     if i % (n_intermed_steps/n_orientations) != 0]
    intermed_angles = [np.pi*1/n_intermed_steps*i for i in intermed_locs]
    # these are the canonical orientations
    freqs = [(f*np.sin(a), f*np.cos(a)) for a, f in itertools.product(ori_angles, base_freqs)]
    # arc, where distance from the origin is half the max (in log space)
    #  skip those values which we've already gotten: 0, pi/4, pi/2, 3*pi/4, and pi
    freqs.extend([(intermed_freq*np.sin(i),
                   intermed_freq*np.cos(i)) for i in intermed_angles])
    if round_flag:
        freqs = np.round(freqs)
    return freqs


def _create_stim(res, freqs, phi, n_exemplars, output_dir, stimuli_name,
                 stimuli_description_csv_name, col_names, stim_type, mask=None):
    """helper function to create the stimuli and and stimuli description csv

    stim_type: {'logpolar', 'constant'}. which type of stimuli to make. determines which function
    to call, gen_log_polar_stim_set or gen_constant_stim_set. if constant, mask must be set
    """
    if stim_type == 'logpolar':
        masked_stim, stim, mask = gen_log_polar_stim_set(res, freqs, phi)
    elif stim_type == 'constant':
        masked_stim, stim = gen_constant_stim_set(res, mask, freqs, phi)
    np.save(os.path.join(output_dir, stimuli_name), stim)

    # log-polar csv
    df = []
    for i, ((w_1, w_2), p) in enumerate(itertools.product(freqs, phi)):
        df.append((w_1, w_2, p, res, i, i / n_exemplars))
    df = pd.DataFrame(df, columns=col_names)
    df.to_csv(os.path.join(output_dir, stimuli_description_csv_name), index=False)
    return stim, mask


def main(pixel_diameter=714, degree_diameter=8.4, n_exemplars=8, n_freq_steps=6,
         n_logpolar_orientations=4, n_constant_orientations=8, n_logpolar_intermed_samples=1,
         n_constant_intermed_samples=0, constant_freq_target_eccen=2.5,
         output_dir="../data/stimuli/", stimuli_name='task-sfp_stimuli.npy',
         mask_json_name="task-sfp_mask.json", mat_file_name="spatialFreqStim.mat",
         stimuli_description_csv_name='task-sfp_stim_description.csv'):
    """create the stimuli for the spatial frequency preferences experiment

    We save the unmasked stimuli, along with a json file that describes the radius we think is
    necessary (in pixels and in degrees) of the anti-aliasing mask.

    Our stimuli are constructed from a 2d frequency space, with w_r on the x-axis and w_a on the
    y. By default, the stimuli we want for our experiment then lie along the x-axis, the y-axis,
    the + and - 45-degree angle lines (that is, x=y and x=-y, y>0 for both), and the arc that
    connects all of them. For those stimuli that lie along straight lines / axes, they'll have
    frequencies from 2^(2.5) to 2^(7) (distance from the radius) sampled `n_freq_steps` times,
    while the arc will lie approximately half-way between the two extremes. We round all
    frequencies to the nearest integer, because non-integer frequencies cause obvious breaks
    (especially in the spirals).

    By changing `n_logpolar_orientations`, you can change the number of "canonical orientations" we
    sample. Similarly, `n_logpolar_intermed_samples` controls the number of intermediate spirals
    between each canonical orientations.

    We also generate constant stimuli whose spatial frequency approximately matches that of our
    logpolar ones near `constant_freq_target_eccen`. They will have the same number of frequency
    steps as the logpolar ones, but can have different numbers of orientations and intermediate
    samples, controlled by `n_constant_orientations` and `n_constant_intermed_samples`.

    there will be `n_exemplars` (default 8) different phases equally spaced from 0 to 2 pi:
    np.array(range(n_exemplars))/n_exemplars.*2*np.pi

    The actual stimuli will be saved as {stimuli_name} in the output_dir. A description of the
    stimuli properties, in the order found in the stimuli, is saved at
    {stimuli_description_csv_name} in the output folder, as a pandas DataFrame. We will also save
    constant stimuli by replacing "task-sfp" with "task-sfpconstant" in stimuli_name (and doing the
    same for stimuli_description_csv_name). Therefore, these strings must contain "task-sfp".

    We also save a .mat file which contains the constant and logpolar stimuli as
    spatialFreqStim.mat in the output_dir. This also contains fields that give the radius we think
    is necessary (in pixels and in degrees) of the anti-aliasing mask.

    returns the logpolar and constant stimuli

    """
    if 'task-sfp' not in stimuli_name:
        raise Exception("stimuli_name must contain task-sfp!")
    if 'task-sfp' not in stimuli_description_csv_name:
        raise Exception("stimuli_description_csv_name must contain task-sfp!")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    base_freqs = [2**i for i in np.linspace(2.5, 7, n_freq_steps, endpoint=True)]
    freqs = _gen_freqs(base_freqs, n_logpolar_orientations, n_logpolar_intermed_samples, True)
    # in order to determine the spatial frequencies for the constant stimuli, we grab the spatial
    # frequencies at different eccentricities for the logpolar stimuli we'll create...
    _, sf_df = check_stim_properties(pixel_diameter, None, degree_diameter, 0,
                                     np.round(base_freqs))
    # we won't have exactly this eccentricity in the dataframe because of how we sample the space,
    # so this grabs the closest one
    sf_df_eccen = sf_df.eccen.unique()[np.abs(sf_df.eccen.unique() -
                                              constant_freq_target_eccen).argmin()]
    # this will then grab the stimuli's spatial frequency at that eccentricity...
    constant_freqs = sf_df[sf_df.eccen == sf_df_eccen].local_freq_cpp.values
    # which we pass to _gen_freqs to get the full set of frequencies (with different orientations)
    constant_freqs = _gen_freqs(constant_freqs, n_constant_orientations,
                                n_constant_intermed_samples, False)
    phi = np.array(range(n_exemplars))/n_exemplars*2*np.pi
    # in case we need to return them
    stim, constant_stim = None, None
    mat_save_dict = {}
    if os.path.isfile(os.path.join(output_dir, stimuli_name.replace('task-sfp',
                                                                    'task-sfpconstant'))):
        raise Exception("stimuli already exists!")
    if os.path.isfile(os.path.join(output_dir,
                                   stimuli_description_csv_name.replace('task-sfp',
                                                                        'task-sfpconstant'))):
        raise Exception("stimuli already exists!")
    if os.path.isfile(os.path.join(output_dir, stimuli_name)):
        raise Exception("stimuli already exists!")
    if os.path.isfile(os.path.join(output_dir, stimuli_description_csv_name)):
        raise Exception("stimuli already exists!")
    # log-polar stimuli and csv
    stim, mask = _create_stim(pixel_diameter, freqs, phi, n_exemplars, output_dir,
                              stimuli_name, stimuli_description_csv_name,
                              ['w_r', 'w_a', 'phi', 'res', 'index', 'class_idx'], 'logpolar')
    json_to_save = {'mask_radius_degrees': find_ecc_range_in_degrees(mask, degree_diameter/2, 0)[0],
                    'mask_radius_pixels': find_ecc_range_in_pixels(mask, 0)[0]}
    with open(os.path.join(output_dir, mask_json_name), 'w') as f:
        json.dump(json_to_save, f)
    mat_save_dict['stimuli'] = stim
    # constant stimuli and csv
    constant_stim, _ = _create_stim(pixel_diameter, constant_freqs, phi, n_exemplars,
                                    output_dir,
                                    stimuli_name.replace('task-sfp', 'task-sfpconstant'),
                                    stimuli_description_csv_name.replace('task-sfp',
                                                                         'task-sfpconstant'),
                                    ['w_x', 'w_y', 'phi', 'res', 'index', 'class_idx'],
                                    'constant', mask)
    mat_save_dict['constant_stimuli'] = constant_stim
    mat_save_dict.update(json_to_save)
    sio.savemat(os.path.join(output_dir, mat_file_name), mat_save_dict)
    return stim, constant_stim


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          argparse.RawDescriptionHelpFormatter):
        pass
    parser = argparse.ArgumentParser(description=(main.__doc__),
                                     formatter_class=CustomFormatter)
    parser.add_argument("--output_dir", '-o', help="directory to place stimuli and indices in",
                        default="data/stimuli")
    parser.add_argument("--stimuli_name", '-n', help="name for the unshuffled stimuli",
                        default="task-sfp_stimuli.npy")
    parser.add_argument("--stimuli_description_csv_name", '-d',
                        help="name for the csv that describes unshuffled stimuli",
                        default="task-sfp_stim_description.csv")
    parser.add_argument("--mask_json_name", '-j',
                        help=("name for the json that contains the radius we think necessary for "
                              "the anti-aliasing mask"),
                        default="task-sfp_mask.json")
    parser.add_argument("--mat_file_name", "-m",
                        help="name for the .mat file that contains the equivalent information",
                        default="spatialFreqStim.mat")
    args = vars(parser.parse_args())
    _ = main(**args)
