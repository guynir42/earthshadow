import datetime
import numpy as np

import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
import astropy.time
from astropy.time import Time
import astropy.units as u

import matplotlib.pyplot as plt

EARTH_RADIUS = 6371 * u.km
LOW_EARTH_ORBIT = 300 * u.km + EARTH_RADIUS
MEDIUM_EARTH_ORBIT = 1000 * u.km + EARTH_RADIUS
GPS_SATELITE_ORBIT = 26600 * u.km
GEOSYNCHRONOUS_ORBIT = 42164 * u.km
HIGH_EARTH_ORBIT = GEOSYNCHRONOUS_ORBIT * 2

DEFAULT_ORBITAL_RADIUS = GEOSYNCHRONOUS_ORBIT


def get_earth_shadow(orbit=None, geocentric_orbit=True, geocentric_angle=True):
    """
    Get the angle of the radius of Earth's shadow,
    where it intercepts the sky at an orbital radius,
    as seen by an observer at the center of the Earth.

    This is the geometric shadow, and does not include
    partial shadowing by the atmosphere.
    The atmosphere allows bending of some light to reach
    an angle of about 1-2 degrees into the geometric shadow.

    When inputting the orbit as a float (assume km) or a Quantity
    it is assumed this value includes the radius of the Earth.
    To give the orbit height above the Earth's surface,
    specify geocentric=False.

    Parameters
    ----------
    orbit: float or astropy.units.Quantity
        The orbital radius of the satellite.
        This is measured from the center of the Earth
        (e.g., LEO would be 200 + 6371 = 6571 km).
        If given as float assume kilometers.
        Defaults to 42164 km (geosynchronous orbit).
    geocentric_orbit: bool
        If True, assume the orbit is given as the
        distance from the center of the Earth (default).
        If False, assume the orbit is given as the
        distance above the Earth's surface.

    Returns
    -------
    angle: astropy.units.Quantity
        The angle of the Earth shadow.
    """
    orbit = interpret_orbit(orbit, geocentric=geocentric_orbit)

    if orbit < EARTH_RADIUS:
        raise ValueError(
            f"Orbit radius {orbit} is below Earth radius {EARTH_RADIUS}. "
            "If you intend to give the orbit height above the Earth surface, "
            "set geocentric_orbit=False."
        )

    angle = np.arcsin(EARTH_RADIUS / orbit).to(u.deg)

    if not geocentric_angle:
        angle = geocentric_to_topocentric_angle(angle, orbit=orbit)

    return angle


def dist_from_shadow_center(
    ra,
    dec,
    time=None,
    orbit=None,
    obs=None,
    geocentric_orbit=True,
    geocentric_output=True,
    verbose=False,
):
    """
    Calculate the angular distance from the center of the shadow
    for the given coordinates at the given time.
    The angle is measured by an observer at the center of the Earth
    (default, but can also choose geocentric_output=False).
    This should be compared to the size of the shadow from the same
    vantage point, as given by the `get_earth_shadow` function.

    For an orbit at infinite height,
    the angular distance is the same as
    the distance from the anti-solar point.
    For a finite orbit radius (e.g., GEO)
    this also incorporates the effects of parallax.

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
    obs: astropy.coordinates.EarthLocation, string, or 3-tuple of (lon, lat, alt)
        The coordinates of the observatory.
        Default is the coordinates of the ZTF observatory.
        If given as string will use the
        `astropy.coordinates.EarthLocation.of_site` method.
        If given as a list or tuple, will assume it is
        (longitude, latitude, altitude) in degrees and meters.

    geocentric_orbit: bool
        If True, assume the orbit is given as the
        distance from the center of the Earth (default).
        If False, assume the orbit is given as the
        distance above the Earth's surface.
    geocentric_output: bool
        Output the angle as measured by an observer at
        the center of the Earth (default).
        If False, will output the angle as measured by
        an observer on the Earth's surface, assuming
        the observer is under the center of the shadow.
    verbose: bool
        Whether to print various intermediate results.

    Returns
    -------
    dist: scalar float or array of floats
        The distance from the center of the shadow in degrees.
        The angle is measured by an observer at the center of the Earth.
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
    time = interpret_time(time)

    # verify/convert the orbit
    orbit = interpret_orbit(orbit, geocentric=geocentric_orbit)

    # verify/convert the observatory coordinates
    if obs is None:
        obs = coord.EarthLocation.of_site("palomar")
    if isinstance(obs, str):
        obs = coord.EarthLocation.of_site(obs)
    if isinstance(obs, (list, tuple)):
        if len(obs) != 3:
            raise ValueError("obs must be a 3-tuple or list of (lon, lat, alt)")
        new_obs = list(obs)
        if not isinstance(new_obs[0], u.quantity.Quantity):
            new_obs[0] = new_obs[0] * u.deg
        if not isinstance(new_obs[1], u.quantity.Quantity):
            new_obs[1] = new_obs[1] * u.deg
        if not isinstance(new_obs[2], u.quantity.Quantity):
            new_obs[2] = new_obs[2] * u.m

        obs = coord.EarthLocation.from_geodetic(
            lon=new_obs[0],
            lat=new_obs[1],
            height=new_obs[2],
        )
    if not isinstance(obs, coord.EarthLocation):
        raise ValueError(
            "obs must be a 3-tuple of (lon, lat, alt) "
            "or a string with a name of a known observatory "
            "or an astropy.coordinates.EarthLocation"
        )

    if verbose:
        print(f"observatory: {obs}")

    # get the anti-sun position in geo-centric coordinates
    anti_sun = get_anti_sun(time)
    if verbose:
        print("anti-sun:", anti_sun)

    # convert to astropy coordinates
    obs_frame = coord.CIRS(obstime=time, location=obs)
    target_coords = SkyCoord(ra, dec, frame=obs_frame)
    if verbose:
        print(f"target_coords = {target_coords}")

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
    range_plus = -term_a + np.sqrt(term_a**2 - term_b)  # km
    range_minus = -term_a - np.sqrt(term_a**2 - term_b)  # km

    prefer_plus_range = abs(range_plus) < abs(range_minus)

    if verbose:
        print(
            f"minus_range= {range_minus}, plus_range= {range_plus}, "
            f"prefer_plus_range= {prefer_plus_range}"
        )

    chosen_range = np.where(prefer_plus_range, range_plus, range_minus)

    x_final = x0 + chosen_range * xt
    y_final = y0 + chosen_range * yt
    z_final = z0 + chosen_range * zt

    if verbose:
        print(
            f"x= {x_final}, y= {y_final}, z= {z_final}, "
            f"R= {np.sqrt(x_final**2 + y_final**2 + z_final**2)}"
        )

    new_target_coords = SkyCoord(
        x=x_final,
        y=y_final,
        z=z_final,
        frame=coord.GCRS(obstime=time),  # geocentric
        representation_type="cartesian",
    )
    new_target_coords.representation_type = "spherical"

    if verbose:
        print(f"new target: {new_target_coords}")

    # now we can compute the distance from the anti-sun
    dist = new_target_coords.separation(anti_sun)

    if not geocentric_output:
        # convert to the distance from the Earth's surface
        dist = geocentric_to_topocentric_angle(dist, orbit)

    if verbose:
        print(f"dist = {dist}")

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
    time = interpret_time(time)
    sun = coord.get_sun(time)
    anti_sun = SkyCoord(
        ra=np.mod(sun.ra + 180 * u.deg, 360 * u.deg),
        dec=-sun.dec,
        frame=coord.GCRS(obstime=time),  # geocentric
    )

    return anti_sun


def get_observer_opposite_sun(time=None, latitude=0, altitude=None):
    """
    Get the geolocation of an observer that is
    right under the anti-sun point.

    Parameters
    ----------
    time: astropy.time.Time
        The time of the observation.
        Defaults to now.
    altitude: astropy.units.Quantity
        The altitude of the observer above sea level.
        Defaults to 0. If given as float assume meters.

    Returns
    -------
    obs: astropy.coordinates.EarthLocation
        The geolocation of an observer that is
        right under the anti-sun point.
    """
    if time is None:
        time = Time.now()

    if altitude is None:
        altitude = 0

    if not isinstance(altitude, u.quantity.Quantity):
        altitude *= u.m

    anti_sun = get_anti_sun(time)
    obs = coord.EarthLocation.from_geodetic(
        lon=anti_sun.ra,
        lat=anti_sun.dec + latitude * u.deg,
        height=altitude,
    )

    return obs


def topocentric_to_geocentric_angle(angle, orbit=None, geocentric_orbit=True):
    """
    Convert the angle above an observer
    on the surface of the Earth to an angle
    seen by an observer at the center of the Earth.
    Assume the angle is seen straight over the observer.

    This calculation uses the sine function,
    which assumes the orbital radius is calculated
    at the edges of a circle with a radius equal to "angle".
    The center of this circle is right above the observer,
    but is at a slightly lower altitude than the orbit given.
    If the center of the circle was at the orbit
    (and the edges higher than the orbit),
    then the calculation should use the tangent function.
    For small angles this makes no difference.

    Parameters
    ----------
    angle: float or astropy.units.Quantity
        The angle above the observer.
        If given as float assume degrees.
    orbit: float or astropy.units.Quantity
        The orbital radius of the satellite.
        If given as float assume kilometers.
        Defaults to 42164 km (geosynchronous orbit).

    Returns
    -------
    angle: astropy.units.Quantity
        The angle above the observer.
    """
    if not isinstance(angle, u.quantity.Quantity):
        angle *= u.deg

    orbit = interpret_orbit(orbit, geocentric=geocentric_orbit)

    return np.arcsin(np.sin(angle) * orbit / (orbit + EARTH_RADIUS)).to(u.deg)


def geocentric_to_topocentric_angle(angle, orbit=None, geocentric_orbit=True):
    """
    Convert the angle above an observer
    at the center of the Earth to an angle
    seen by an observer on the surface of the Earth.
    Assume the angle is seen straight over the observer.

    This calculation uses the sine function,
    which assumes the orbital radius is calculated
    at the edges of a circle with a radius equal to "angle".
    The center of this circle is right above the observer,
    but is at a slightly lower altitude than the orbit given.
    If the center of the circle was at the orbit
    (and the edges higher than the orbit),
    then the calculation should use the tangent function.
    For small angles this makes no difference.

    Parameters
    ----------
    angle: float or astropy.units.Quantity
        The angle above the observer.
        If given as float assume degrees.
    orbit: float or astropy.units.Quantity
        The orbital radius of the satellite.
        If given as float assume kilometers.
        Defaults to 42164 km (geosynchronous orbit).

    Returns
    -------
    angle: astropy.units.Quantity
        The angle above the observer.
    """
    if not isinstance(angle, u.quantity.Quantity):
        angle *= u.deg

    orbit = interpret_orbit(orbit, geocentric=geocentric_orbit)

    return np.arcsin(np.sin(angle) * (orbit + EARTH_RADIUS) / orbit).to(u.deg)


def interpret_time(time=None):
    """
    Interpret the user input for time
    in various formats and return it
    as an astropy.time.Time object.

    If given as None (or not given)
    will return the current time.
    Can also can use the string "now".
    """
    if time is None:
        time = Time.now()
    if isinstance(time, str) and time.lower() == "now":
        time = Time.now()
    if isinstance(time, str):
        time = Time(time)
    if isinstance(time, float):  # assume JD??
        time = Time(time, format="jd", scale="utc")
    if isinstance(time, datetime.datetime):
        time = Time(time)

    return time


def interpret_orbit(orbit=None, geocentric=True):
    """
    Convert user inputs for the orbital radius
    into a quantity with units of kilometers.

    Parameters
    ----------
    orbit: float or astropy.units.Quantity
        The orbital radius of the satellite.
        If given as float assume kilometers.
    geocentric: bool
        If True, assume the orbit is measured
        from the center of the Earth (default).
        To measure the orbit height above the
        Earth's center use geocentric=False.

    Returns
    -------
    orbit: astropy.units.Quantity
        The orbital radius of the satellite.
    """
    if orbit is None:
        orbit = DEFAULT_ORBITAL_RADIUS

    if isinstance(orbit, str):
        orbit = orbit.lower().replace(" ", "").replace("_", "").replace("-", "")
        if orbit in ("geosynchronous", "geosync", "geo"):
            orbit = DEFAULT_ORBITAL_RADIUS
        elif orbit in ("lowearthorbit", "lowearth", "leo"):
            orbit = LOW_EARTH_ORBIT
        elif orbit in ("mediumearthorbit", "mediumearth", "meo"):
            orbit = MEDIUM_EARTH_ORBIT
        elif orbit in ("highearthorbit", "highearth", "heo"):
            orbit = HIGH_EARTH_ORBIT
        elif orbit in ("groundpositioningsatellite", "groundpositioning", "gps"):
            orbit = GPS_SATELITE_ORBIT

    if not isinstance(orbit, u.quantity.Quantity):
        orbit *= u.km

    if not geocentric:
        orbit += EARTH_RADIUS

    return orbit


if __name__ == "__main__":
    time = Time("2023-09-23T06:50:00")

    anti = get_anti_sun(time)
    obs = get_observer_opposite_sun(time)
    ret = dist_from_shadow_center(
        ra=anti.ra + 0.0 * u.deg,
        dec=anti.dec + 5.0 * u.deg,
        time=time,
        obs=(0, 0, 0),
        verbose=True,
        orbit=None,
    )
