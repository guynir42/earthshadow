import pytest
import numpy as np

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

from earthshadow import *


def test_user_inputs():
    # test the time inputs
    t0 = Time.now()
    t1 = datetime.datetime.utcnow()

    # assume if the input to get_anti_sun is equivalent,
    # the output will be equivalent
    a1 = get_anti_sun(t0)  # should work as is
    a2 = get_anti_sun(t0.jd)  # assume float is JD
    a3 = get_anti_sun()  # default should be now
    a4 = get_anti_sun(t1)  # datetime should work

    assert a1.separation(a2).to(u.arcsec) < 1 * u.arcsec
    assert a1.separation(a3).to(u.arcsec) < 1 * u.arcsec
    assert a1.separation(a4).to(u.arcsec) < 1 * u.arcsec

    # test the orbit inputs
    a1 = geocentric_to_topocentric_angle(5)  # default to GEO?
    a2 = geocentric_to_topocentric_angle(5, orbit=42164)
    a3 = geocentric_to_topocentric_angle(5, orbit=42164 * u.km)
    a4 = geocentric_to_topocentric_angle(5, orbit="GEO")

    assert isinstance(a1, u.Quantity)
    assert a1.unit == "deg"
    assert np.isclose(a1.value, a2.value)
    assert np.isclose(a1.value, a3.value)
    assert np.isclose(a1.value, a4.value)

    # test the RA/Dec inputs
    ra = 10
    dec = 20
    a1 = dist_from_shadow_center(ra, dec)
    a2 = dist_from_shadow_center(ra * u.deg, dec * u.deg)
    a3 = dist_from_shadow_center([ra], [dec])
    a4 = dist_from_shadow_center(np.array([ra]), np.array([dec]))
    a5 = dist_from_shadow_center(np.array([ra]) * u.deg, np.array([dec]) * u.deg)

    assert abs(a1 - a2) < 0.01 * u.deg
    assert abs(a1 - a3) < 0.01 * u.deg
    assert abs(a1 - a4) < 0.01 * u.deg
    assert abs(a1 - a5) < 0.01 * u.deg

    # test the observatory inputs
    b1 = dist_from_shadow_center(ra, dec, obs="Palomar")
    b2 = dist_from_shadow_center(ra, dec, obs=(-116.863, 33.356, 1700))

    assert abs(a1 - b1) < 0.01 * u.deg
    assert abs(a1 - b2) < 0.01 * u.deg

    # this should be a different place
    c1 = dist_from_shadow_center(ra, dec, obs="Paranal Observatory")
    assert abs(a1 - c1) > 1 * u.deg


def test_earth_shadow_sizes():
    # default value should be GEO
    a = get_earth_shadow()
    assert abs(a.value - 8.7) < 0.1

    a1 = get_earth_shadow("GEO")
    assert abs(a1 - a) < 0.1 * u.deg

    # this is also close to GEO
    a2 = get_earth_shadow(42000)
    assert abs(a2 - a) < 0.1 * u.deg

    # this is about twice Earth's radius
    a3 = get_earth_shadow(13000)
    assert a3 > 2 * a

    # this is lower than Earth's radius
    with pytest.raises(ValueError) as e:
        get_earth_shadow(6000)
        assert "is below Earth radius" in str(e)

    # should work if geocentric_orbit=False
    a4 = get_earth_shadow(6000, geocentric_orbit=False)

    assert a4 > 3 * a

    # what about low Earth orbit?
    a5 = get_earth_shadow(300, geocentric_orbit=False)
    assert a5 > 5 * a

    # make sure the keywords work in geocentric too
    a6 = get_earth_shadow("LEO", geocentric_orbit=True)
    assert abs(a6 - a5) < 0.1 * u.deg


def test_observer_under_shadow():
    time = Time("2022-09-21T00:00:00")  # autumn equinox
    anti = get_anti_sun(time)
    obs = get_observer_opposite_sun(time)
    ret = dist_from_shadow_center(ra=anti.ra, dec=anti.dec, time=time, obs=obs)

    assert ret < 0.1 * u.deg

    offset = 10 * u.deg
    ret = dist_from_shadow_center(ra=anti.ra + offset, dec=anti.dec, time=time, obs=obs)

    # return value is measured from the center of the Earth,
    # while the offset angle is measured by a topocentric observer!

    assert abs(ret.deg - 10) < 0.1


def test_topocentric_to_geocentric():
    angle = 5

    assert abs(4.31 - topocentric_to_geocentric_angle(angle).value) < 0.1
    assert abs(5.75 - geocentric_to_topocentric_angle(angle).value) < 0.1

    # check we get the same results with explicitly passing the orbit
    assert abs(4.31 - topocentric_to_geocentric_angle(angle, orbit=42000).value) < 0.1
    assert abs(5.75 - geocentric_to_topocentric_angle(angle, orbit=42000).value) < 0.1

    # check it works for low orbit (about 1 Earth radius)
    assert abs(3.35 - topocentric_to_geocentric_angle(angle, orbit=13000).value) < 0.1
    assert abs(7.46 - geocentric_to_topocentric_angle(angle, orbit=13000).value) < 0.1

    # check the angle doesn't change for very high orbits
    assert abs(5.0 - topocentric_to_geocentric_angle(angle, orbit=1e6).value) < 0.1
    assert abs(5.0 - geocentric_to_topocentric_angle(angle, orbit=1e6).value) < 0.1
