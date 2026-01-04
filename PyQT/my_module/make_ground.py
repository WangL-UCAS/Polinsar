# -*- coding: utf-8 -*-

import collections
import time

import numpy as np


def groundsolver(gamma, kz=None, groundmag=None, gammavol=None,
                 returnall=False, silent=False):
    if not silent:
        print('kapok.topo.groundsolver | Solving for ground coherence. (' + time.ctime() + ')')
    # Get the two possible ground coherence solutions.
    solutions = linefit(gamma, groundmag)

    if kz is not None:
        if not isinstance(kz, (collections.abc.Sequence, np.ndarray)):
            kz = np.ones((gamma.shape[1], gamma.shape[2]), dtype='float32') * kz

        # Get the volume-dominated coherences corresponding to each ground solution. (Observed coherence farthest from ground.)
        gammav = gamma.copy()
        gammav[0] = np.where(np.abs(solutions[0] - gamma[0]) > np.abs(solutions[0] - gamma[1]), gamma[0], gamma[1])
        gammav[1] = np.where(np.abs(solutions[1] - gamma[0]) > np.abs(solutions[1] - gamma[1]), gamma[0], gamma[1])

        # Angular separation between volume coherence and ground -- is it same sign as kz?
        sep = np.angle(gammav * np.conj(solutions)) * np.sign(kz)

        ground = np.where(sep[0] >= 0, solutions[0], solutions[1])
        groundalt = np.where(sep[0] >= 0, solutions[1], solutions[0])
        volindex = (np.abs(gamma[1] - ground) > np.abs(gamma[0] - ground))
    elif gammavol is not None:
        # Of the two observed coherences, assume the volume-dominated coherence is the one
        # which is closer to the input gammavol array.
        volindex = (np.abs(gamma[1] - gammavol) < np.abs(gamma[0] - gammavol))
        gammav = np.where(volindex, gamma[1], gamma[0])

        # Choose the ground that is farther from gammav.
        ground = np.where(np.abs(gammav - solutions[0]) > np.abs(gammav - solutions[1]), solutions[0], solutions[1])
        groundalt = np.where(np.abs(gammav - solutions[0]) > np.abs(gammav - solutions[1]), solutions[1], solutions[0])
    else:
        print(
            'kapok.topo.groundsolver | Neither kz or estimated volume coherence specified.  Unable to choose between ambiguous ground solutions.  Aborting.')
        ground = None
        groundalt = None
        volindex = None

    if not silent:
        print('kapok.topo.groundsolver | Complete. (' + time.ctime() + ')')

    if returnall:
        return ground, groundalt, volindex
    else:
        return ground


def linefit(gamma, groundmag=None):

    if groundmag is None:
        groundmag = np.ones((gamma.shape[1], gamma.shape[2]), dtype='float32')
    elif not isinstance(groundmag, (collections.abc.Sequence, np.ndarray)):
        groundmag = np.ones((gamma.shape[1], gamma.shape[2]), dtype='float32') * groundmag

    groundmag[groundmag > 1] = 1.0

    solutions = np.zeros(gamma.shape, dtype='complex64')

    # Intersections between line through gamma and circle with radius groundmag:
    a = np.square(np.abs(gamma[0] - gamma[1]))
    b = 2 * np.real(gamma[0] * np.conj(gamma[1])) - 2 * np.square(np.abs(gamma[1]))
    c = np.square(np.abs(gamma[1])) - np.square(np.abs(groundmag))

    xa = (-1 * b - np.sqrt(np.square(b) - 4 * a * c)) / (2 * a)
    xb = (-1 * b + np.sqrt(np.square(b) - 4 * a * c)) / (2 * a)

    solutions[0] = xa * gamma[0] + (1 - xa) * gamma[1]
    solutions[1] = xb * gamma[0] + (1 - xb) * gamma[1]

    # Is the coherence magnitude given by groundmag lower than both observed
    # coherences? (e.g., no valid intersection)
    ind = (groundmag < np.abs(gamma[0])) & (groundmag < np.abs(gamma[1]))
    if np.any(ind):
        solutions[0][ind] = np.nan
        solutions[1][ind] = np.nan

    # Are any of the solutions within the observed coherence region?
    ind = np.sign(np.angle(solutions * np.conj(gamma[1]))) == np.sign(np.angle(gamma[0] * np.conj(solutions)))
    if np.any(ind):
        solutions[ind] = np.nan

    # Both solutions invalid:
    ind = np.isnan(solutions[0]) & np.isnan(solutions[1])
    if np.any(ind):
        solutions[0][ind] = gamma[0][ind]
        solutions[1][ind] = gamma[1][ind]

    # First solution invalid.
    ind = np.isnan(solutions[0]) & np.isfinite(solutions[1])
    if np.any(ind):
        gammareplace = np.where(np.abs(solutions[1] - gamma[0]) > np.abs(solutions[1] - gamma[1]), gamma[0], gamma[1])
        solutions[0][ind] = gammareplace[ind]

    # Other solution invalid.
    ind = np.isnan(solutions[1]) & np.isfinite(solutions[0])
    if np.any(ind):
        gammareplace = np.where(np.abs(solutions[0] - gamma[0]) > np.abs(solutions[0] - gamma[1]), gamma[0], gamma[1])
        solutions[1][ind] = gammareplace[ind]

    return solutions