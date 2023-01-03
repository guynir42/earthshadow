import pytest
import numpy as np
import datetime
import astropy.coordinates as coord
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

from src.earthshadow import *


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


def test_shadow_center():
    time = Time("2023-09-23T00:00:00")  # equinox
    anti = get_anti_sun(time, orbit="GEO")
    obs = get_observer_opposite_sun(time)

    # this should be the same as the anti-sun
    a1 = get_shadow_center(time=time, obs=obs, orbit="GEO")

    assert anti.separation(a1) < 0.2 * u.deg

    time = Time("2021-05-12T18:00:00")  # random time
    anti = get_anti_sun(time, orbit="GEO")
    obs = get_observer_opposite_sun(time)

    # this should be the same as the anti-sun
    a2 = get_shadow_center(time=time, obs=obs, orbit="GEO")

    assert anti.separation(a2) < 0.2 * u.deg

    # now get an observer at a different location:
    time = Time("2021-09-29T08:00:00")  # equinox, midnight at Palomar
    anti = get_anti_sun(time, orbit="GEO")
    obs = "palomar"

    # this should be offset because of parallax
    a3 = get_shadow_center(time=time, obs=obs, orbit="GEO")

    assert abs(anti.separation(a3) - 5 * u.deg) < 0.1 * u.deg


def test_earth_shadow_sizes():
    # default value should be GEO
    a = get_shadow_radius()
    assert abs(a.value - 8.7) < 0.1

    a1 = get_shadow_radius("GEO")
    assert abs(a1 - a) < 0.1 * u.deg

    # this is also close to GEO
    a2 = get_shadow_radius(42000)
    assert abs(a2 - a) < 0.1 * u.deg

    # this is about twice Earth's radius
    a3 = get_shadow_radius(13000)
    assert a3 > 2 * a

    # this is lower than Earth's radius
    with pytest.raises(ValueError) as e:
        get_shadow_radius(6000)
        assert "is below Earth radius" in str(e)

    # should work if geocentric_orbit=False
    a4 = get_shadow_radius(6000, geocentric_orbit=False)

    assert a4 > 3 * a

    # what about low Earth orbit?
    a5 = get_shadow_radius(300, geocentric_orbit=False)
    assert a5 > 5 * a

    # make sure the keywords work in geocentric too
    a6 = get_shadow_radius("LEO", geocentric_orbit=True)
    assert abs(a6 - a5) < 0.1 * u.deg


def test_observer_under_shadow():
    time = Time("2021-09-23T12:00:00")  # autumn equinox
    anti = get_anti_sun(time)
    obs = get_observer_opposite_sun(time)

    aa = coord.AltAz(location=obs, obstime=time)
    anti_aa = anti.transform_to(aa)

    assert 90 * u.deg - anti_aa.alt < 1 * u.deg  # should be close to 90 degrees

    ret = dist_from_shadow_center(
        ra=anti.ra, dec=anti.dec, time=time, obs=obs, verbose=0
    )

    assert ret < 0.1 * u.deg

    # move the target on the declination axis
    offset = 5 * u.deg
    ret = dist_from_shadow_center(ra=anti.ra, dec=anti.dec + offset, time=time, obs=obs)

    # measured by observer at center of Earth:
    assert abs(ret - offset * 0.87) < 0.2 * u.deg

    ret = dist_from_shadow_center(
        ra=anti.ra, dec=anti.dec + offset, time=time, obs=obs, geocentric_output=False
    )
    assert abs(ret - offset) < 0.2 * u.deg  # measured by observer on the surface

    # move the target on the RA axis a bit
    ret = dist_from_shadow_center(
        ra=anti.ra + offset, dec=anti.dec, time=time, obs=obs, verbose=1
    )
    assert (
        abs(ret / np.cos(anti.dec) - offset * 0.87) < 0.2 * u.deg
    )  # measured by observer at center of Earth

    # now return a value measured from the Earth's surface
    ret = dist_from_shadow_center(
        ra=anti.ra + offset,
        dec=anti.dec,
        time=time,
        obs=obs,
        verbose=1,
        geocentric_output=False,
    )
    assert (
        abs(ret / np.cos(anti.dec) - offset) < 0.2 * u.deg
    )  # measured by observer on the surface

    # now move the observer to a higher latitude
    obs = get_observer_opposite_sun(time, latitude=30)
    ret = dist_from_shadow_center(
        ra=anti.ra, dec=anti.dec, time=time, obs=obs, geocentric_output=False
    )

    # the result should be offset by a parallax of about 1/2 Earth radius
    # at the distance of 42000-6400 km (GEO) this would be about 5 degrees
    assert abs(ret - 5 * u.deg) < 0.1 * u.deg

    # try it for a lower orbit
    ret = dist_from_shadow_center(
        ra=anti.ra,
        dec=anti.dec,
        time=time,
        obs=obs,
        orbit="GPS",
        geocentric_output=False,
    )

    # the parallax should be substantially larger
    assert ret > 8 * u.deg


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
