import datetime
import numpy as np

import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
import astropy.time
from astropy.time import Time
import astropy.units as u

import matplotlib.pyplot as plt

# just for fun let's use ZTF's coordinates
DEFAULT_OBS_LONGITUDE = 116.8356 * u.deg
DEFAULT_OBS_LATITUDE = 33.3634 * u.deg
DEFAULT_OBS_ALTITUDE = 1.872 * u.km

DEFAULT_ORBITAL_RADIUS = 42164 * u.km


def dist_from_shadow_center(ra, dec, time=None, orbit=None, obs=None, verbose=False):
    """
    Calculate the distance from the center of the shadow
    for the given coordinates at the given time.

    Parameters
    ----------
    ra: scalar float or array of floats
        The right ascension of the object(s) in degrees.
    dec: scalar float or array of floats
        The declination of the object(s) in degrees.
    time: scalar float or astropy.time.Time
        The time of the observation.
        Default is the current time.
    orbit: scalar float
        The orbital radius at which we assume the targets are moving.
        The lower the orbit, the bigger the Earth's shadow will be.
        Default is 42164 km, which is the altitude of geosynchronous satellites.
    obs: astropy.coordinates.SkyCoord or 3-tuple of (lon, lat, alt)
        The coordinates of the observatory.
        Default is the coordinates of the ZTF observatory.
    verbose: bool
        Whether to print various intermediate results.

    Returns
    -------
    dist: scalar float or array of floats
        The distance from the center of the shadow in degrees.
         (output is same size as ra/dec)
    """

    # verify/convert the coordinates
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)

    if ra.shape != dec.shape:
        raise ValueError("ra and dec must have the same shape")

    if not isinstance(ra, u.quantity.Quantity):
        ra *= u.deg
    if not isinstance(dec, u.quantity.Quantity):
        dec *= u.deg

    # verify/convert the time
    if time is None:
        time = Time.now()
    if isinstance(time, float):  # assume JD??
        time = Time(time, format="jd")
    if isinstance(time, datetime.datetime):
        time = Time(time)

    # verify/convert the orbit
    if orbit is None:
        orbit = DEFAULT_ORBITAL_RADIUS

    if not isinstance(orbit, u.quantity.Quantity):
        orbit *= u.km

    # verify/convert the observatory coordinates
    if obs is None:
        obs = coord.EarthLocation.from_geodetic(
            lon=DEFAULT_OBS_LONGITUDE,
            lat=DEFAULT_OBS_LATITUDE,
            height=DEFAULT_OBS_ALTITUDE,
        )
    if isinstance(obs, tuple):
        obs = coord.EarthLocation.from_geodetic(
            lon=obs[0],
            lat=obs[1],
            height=obs[2],
        )
    if not isinstance(obs, coord.EarthLocation):
        raise ValueError(
            "obs must be a 3-tuple of (lon, lat, alt) "
            "or an astropy.coordinates.EarthLocation"
        )

    # get the anti-sun position in geo-centric ecliptic coordinates
    anti_sun = get_anti_sun(time)
    if verbose:
        print("anti-sun:", anti_sun)

    # convert to astropy coordinates
    target_coords = SkyCoord(ra, dec, frame="geocentrictrueecliptic", obstime=time)

    # we can express these coordinates as the observer location x0, y0, z0 plus
    # a vector t that points from the observer to the target (xt, yt, zt)
    # so (x,y,z) = (x0,y0,z0) + t * (xt,yt,zt)
    xt = target_coords.cartesian._values["x"]
    yt = target_coords.cartesian._values["y"]
    zt = target_coords.cartesian._values["z"]
    if verbose:
        print(f"xt= {xt}, yt= {yt}, zt= {zt}")

    x0 = obs.to_geocentric()[0]
    y0 = obs.to_geocentric()[1]
    z0 = obs.to_geocentric()[2]
    if verbose:
        print(f"x0= {x0}, y0= {y0}, z0= {z0}")

    # this vector will intersect the orbital radius R when
    # (x0 + r*xt)^2 + (y0 + r*yt)^2 + (z0 + r*zt)^2 = R^2
    # the solution is r = (x0*xt+y0+xt+z0*zt) +/- sqrt((x0*xt+y0+xt+z0*zt)^2 - (x0^2+y0^2+z0^2-R^2))
    # which gives the range from observer to target
    term_a = x0 * xt + y0 * yt + z0 * zt
    term_b = x0**2 + y0**2 + z0**2 - orbit**2
    range_plus = term_a + np.sqrt(term_a**2 - term_b)  # km
    range_minus = term_a - np.sqrt(term_a**2 - term_b)  # km

    prefer_plus_range = abs(range_plus) < abs(range_minus)

    if verbose:
        print(
            f"minus_range= {range_minus}, plus_range= {range_plus}, "
            f"prefer_plus_range= {prefer_plus_range}"
        )

    range = np.where(prefer_plus_range, range_plus, range_minus)

    new_target_coords = SkyCoord(
        x=x0 + range * xt,
        y=y0 + range * yt,
        z=z0 + range * zt,
        frame="geocentrictrueecliptic",
        obstime=time,
        representation_type="cartesian",
    ).transform_to("icrs")

    if verbose:
        print(f"new target: {new_target_coords}")

    # now we can compute the distance from the anti-sun
    dist = new_target_coords.separation(anti_sun)

    return dist


def get_anti_sun(time=None):
    """
    Get the anti-sun position in geo-centric ecliptic coordinates

    Parameters
    ----------
    time: astropy.time.Time
        The time of the observation.
        Defaults to now.
    Returns
    -------
    anti_sun: astropy.coordinates.SkyCoord
        The anti-sun position.
    """
    sun = coord.get_sun(time)
    anti_sun = SkyCoord(
        ra=np.mod(sun.ra + 180 * u.deg, 360 * u.deg),
        dec=-sun.dec,
    )

    return anti_sun


def get_observer_opposite_sun(time=None, altitutde=None):
    """
    Get the geolocation of an observer that is
    right under the anti-sun point.

    Parameters
    ----------
    time: astropy.time.Time
        The time of the observation.
        Defaults to now.
    Returns
    -------
    obs: astropy.coordinates.EarthLocation
        The geolocation of an observer that is
        right under the anti-sun point.
    """
    if time is None:
        time = Time.now()

    if altitutde is None:
        altitutde = DEFAULT_OBS_ALTITUDE

    anti_sun = get_anti_sun(time)
    obs = coord.EarthLocation.from_geodetic(
        lon=anti_sun.ra,
        lat=anti_sun.dec,
        height=altitutde,
    )

    return obs


if __name__ == "__main__":
    time = Time("2022-09-21T00:00:00")

    anti = get_anti_sun(time)
    obs = get_observer_opposite_sun(time)
    ret = dist_from_shadow_center(
        ra=anti.ra, dec=anti.dec, time=time, obs=obs, verbose=True
    )
    print(ret)
