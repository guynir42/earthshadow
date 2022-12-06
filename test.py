import pytest
import numpy as np

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

from earthshadow import get_anti_sun, get_observer_opposite_sun, dist_from_shadow_center


def test_observer_under_shadow():
    time = Time("2022-09-21T00:00:00")  # autumn equinox
    anti = get_anti_sun(time)
    obs = get_observer_opposite_sun(time)
    ret = dist_from_shadow_center(ra=anti.ra, dec=anti.dec, time=time, obs=obs)

    assert ret < 0.1 * u.deg
