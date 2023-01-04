# earthshadow

An astropy based package to calculate if objects are in Earth's shadow

## Introduction

The Earth casts a shadow roughly the shape of a cylinder into space,
with a radius equal to the Earth's radius, around the point opposite
the sun (the "anti-solar point").
Three things make it non-trivial to calculate if an object in space is
inside the shadow:

- The shadow's angular size depends on the distance of the object from
  the center of the Earth (or the altitude of the object above the surface).
  E.g., for a Low Earth Orbit (LEO) satellite, most of the sky is in shadow,
  but for a geostationary (GEO) satellite, the angular radius is about 9 degrees.
- The exact center and extent of the shadow depend on the observer's position,
  because of parallax.
- The shadow does not have a sharp edge, because the Earth's atmosphere refracts
  some of the light into the geometric shadow region. This allows some light to penetrate
  about 1 or 2 degrees into the shadow.

The reason this is important is that satellites tend to reflect sunlight,
either constant (diffuse reflection) so they tend to leave streaks in
astronomical images (e.g., see [Nir et al. 2018](https://ui.adsabs.harvard.edu/abs/2018AJ....156..229N/abstract))
or by specular reflection causing glints (e.g., see [Nir et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.2477N/abstract)).
Knowing that a point in the sky is in the shadow of the Earth allows
one to rule out a source of light as coming from a satellite in Earth's orbit (up to some altitude).
Because the vast majority of satellites are in LEO or GEO,
it is usually good enough to to test this for GEO altitude.

## Installation

Just `pip install earthshadow`.
There are no exceptional dependencies other than astropy.

## Usage

The two easiest functions to use are `get_shadow_center` and `get_shadow_radius`.
These will tell you the position and radius of the shadow in the sky,
allowing, e.g., to plan observations in a certain direction to purposefully
take images in the shadow (where there are no satellite streaks/glints).

```python
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from astropy import units as u
    from earthshadow import get_shadow_center, get_shadow_radius

    # Get the center of the shadow in the sky
    center = get_shadow_center(Time.now(), location='APO', orbit='GEO')

    # Get the radius of the shadow in the sky
    radius = get_shadow_radius(orbit='GEO')
```

The center of the shadow is essentially the anti-sun position,
shifted by parallax due to the observer's position.
It is returned as a SkyCoord object, to be used,
e.g., for targeting a telescope in that direction.
The shadow radius is just the angular size (in degrees)
given the orbit of the target. Note that this is includes
the full geometric shadow, not accounting for atmospheric refraction.
The full shadow is generally 1--2 degrees smaller than this.

Note that by default the `get_shadow_radius` function will return
the size of the shadow as seen **from the center of the Earth**,
as this is most useful for internal calculations in the module.
To translate this angle to the size of the shadow as seen by an observer
at sea-level, use `geocentric_angle = False` in the function call.

Another use-case is to check if a given object is in the shadow.
This is usually used after already taking observations,
to rule out that a streak/glint is due to a satellite.
Use `dist_from_shadow_center` to get the angle (in degrees)
the point (or points) are from the shadow's center,
and compare that to the output of `get_shadow_radius`.
Both are given, by default, as angles seen by an observer
at the center of the Earth.
If the distance is more than 2 degrees smaller than the shadow,
the source can not be a satellite reflecting sunlight.
Note that the source can still be an object in very high orbit
or a light emitting object (e.g., a laser shot down from a satellite).

For example, given two arrays of coordinates given by `ra_values` and `dec_values`
and a single observation time given by `time`:

```python
    dist = dist_from_shadow_center(ra_values, dec_values, time=time, location='Palomar', orbit='GEO')

    # Check if the object is in the shadow
    for i, d in enumerate(dist):
        if d < radius - 2*u.deg:
            print(f'Object ({ra_values[i]}, {dec_values[i]}) is in the shadow')
        else:
            print(f'Object ({ra_values[i]}, {dec_values[i]}) is not in the shadow')
```

To make plots of the shadow region we provide two methods:
The `show_shadow_region` which will display a rectangular map
of the RA/Dec region around the anti-sun position,
that shows, for a given orbit, what the shadow looks like
after applying the parallax appropriate for that observer.
The `show_skymap` will show the position of the shadow
on an aitoff projection, with some other useful mapping aids
(to be added...).

```python
    show_shadow_region(time='now', location='Cerro Paranal', orbit='GEO')
    show_skymap('now', location='Green Bank Telescope', orbit='Medium')
```

### Input formatting

There are a few ways to format the inputs to various functions
in this package.

- Time: input a string in ISO format, e.g., `'2021-01-01 00:00:00'`,
  or an astropy Time object. Also, can simply use the string "now".
  If given as a float, assumes it is a Julian Date.
  Can also input a standard datetime object.
- Location (`obs`): use a string name of the observatory (e.g., 'palomar')
  or give the earth location as an astropy EarthLocation object, or as
  a tuple of three numbers: (longitude, latitude, altitude).
  The numbers can be given as astropy Quantity objects, or as numbers,
  in which case they are assumed to be in degrees and meters, respectively.
- Orbit (`orbit`): use a string name of the orbit
  (one of: 'LEO', 'GEO', 'GPS', 'MEDIUM', or 'HIGH')
  or give the radius of the orbit as a number in kilometers. This assumes the orbit
  distance is measured from the center of the Earth. If instead you'd like to
  specify the altitude above sea-level, use the additional input
  `geocentric_orbit = False` which means the Earth's radius is subtracted
  from the input to `orbit` in km.

The default values are given at the top of the `earthshadow.py` module,
and can always be accessed by passing a `None` value to any of these inputs.
Unless edited by the user, these defaults are time='now', obs='Palomar',
and orbit='GEO'.
The medium Earth orbit is arbitrarily defined as 1000 (which is the upper end of LEO, really)
and the high Earth orbit is arbitrarily defined as twice the GEO orbital radius.
Without a better defition of "high orbit", this is a good estimate of where
high satellites will orbit. Above this altitude the Earth's shadow is smaller than
the atmospheric refraction, so there is no real place where a satellite is in complete shadow
(e.g., the Moon is never completely darkened by the Earth, even in a total lunar eclipse).

### Additional notes

To be added...
