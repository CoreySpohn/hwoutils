"""Focused tests for Gregorian calendar dates converted to Julian days."""

import jax
import jax.numpy as jnp
import pytest

from hwoutils import conversions as conv


@pytest.fixture(autouse=True)
def enable_x64():
    """Preserve fractional days at the magnitude of Julian dates."""
    with jax.enable_x64():
        yield


class TestGregorianToJd:
    """Verify reference epochs, calendar arithmetic, and array inputs."""

    def test_j2000_epoch(self):
        """J2000.0 falls at noon on January 1, 2000."""
        # Meeus, Astronomical Algorithms (2nd ed.), ch. 7.
        assert conv.gregorian_to_jd(2000, 1, 1.5) == 2451545.0

    def test_unix_epoch(self):
        """The Unix epoch falls at midnight on January 1, 1970."""
        # Follows from the Meeus (2nd ed.) ch. 7 algorithm.
        assert conv.gregorian_to_jd(1970, 1, 1.0) == 2440587.5

    def test_gregorian_calendar_start(self):
        """The first date of the Gregorian reform has the expected JD."""
        # Meeus, Astronomical Algorithms (2nd ed.), ch. 7.
        assert conv.gregorian_to_jd(1582, 10, 15.0) == 2299160.5

    def test_mjd_zero_point(self):
        """MJD zero falls at midnight on November 17, 1858."""
        # Meeus, Astronomical Algorithms (2nd ed.), ch. 7 (MJD = JD - 2400000.5).
        assert conv.gregorian_to_jd(1858, 11, 17.0) == 2400000.5

    def test_fractional_days(self):
        """Exactly representable day fractions shift JD by the same fraction."""
        for year, month, day in [(1970, 1, 1), (2000, 2, 29), (2024, 12, 31)]:
            midnight = conv.gregorian_to_jd(year, month, day)
            for fraction in [0.125, 0.25, 0.5, 0.75]:
                assert (
                    conv.gregorian_to_jd(year, month, day + fraction) - midnight
                    == fraction
                )

    def test_consecutive_days(self):
        """Successive dates within a month are exactly one Julian day apart."""
        for day in range(1, 31):
            assert (
                conv.gregorian_to_jd(2024, 7, day + 1)
                - conv.gregorian_to_jd(2024, 7, day)
                == 1.0
            )

    def test_month_boundaries(self):
        """Thirty-day and thirty-one-day months join without a gap."""
        for month, last_day in [(1, 31), (4, 30), (7, 31), (11, 30)]:
            assert (
                conv.gregorian_to_jd(2024, month + 1, 1)
                - conv.gregorian_to_jd(2024, month, last_day)
                == 1.0
            )

    def test_year_boundaries(self):
        """December 31 and January 1 are exactly one day apart."""
        for year in [1899, 1900, 1999, 2000, 2023, 2024]:
            assert (
                conv.gregorian_to_jd(year + 1, 1, 1)
                - conv.gregorian_to_jd(year, 12, 31)
                == 1.0
            )

    def test_non_leap_february(self):
        """Common years, including 1900, have no February 29."""
        for year in [1900, 2023, 2100]:
            assert (
                conv.gregorian_to_jd(year, 3, 1) - conv.gregorian_to_jd(year, 2, 28)
                == 1.0
            )

    def test_leap_february(self):
        """Leap years, including 2000, have two days after February 28."""
        for year in [2000, 2024, 2400]:
            feb28 = conv.gregorian_to_jd(year, 2, 28)
            feb29 = conv.gregorian_to_jd(year, 2, 29)
            mar1 = conv.gregorian_to_jd(year, 3, 1)
            assert feb29 - feb28 == 1.0
            assert mar1 - feb29 == 1.0
            assert mar1 - feb28 == 2.0

    def test_year_length(self):
        """January-to-January differences agree with days_in_year."""
        years = jnp.arange(1582, 2401)
        lengths = conv.gregorian_to_jd(years + 1, 1, 1) - conv.gregorian_to_jd(
            years, 1, 1
        )
        assert jnp.array_equal(lengths, conv.days_in_year(years))

    def test_vectorized_inputs(self):
        """Arrays of mixed dates produce exactly the scalar conversion values."""
        dates = [(1582, 10, 15.0), (1900, 3, 1.25), (2000, 2, 29.5), (2024, 1, 1.75)]
        years, months, days = (jnp.array(values) for values in zip(*dates, strict=True))
        actual = conv.gregorian_to_jd(years, months, days)
        expected = jnp.array([conv.gregorian_to_jd(*date) for date in dates])
        assert actual.shape == expected.shape
        assert jnp.array_equal(actual, expected)

    def test_astropy_crosscheck(self):
        """Modern dates agree with Astropy's UTC Julian dates."""
        # UTC .jd is a quasi-JD (leap-second days stretch); none of these dates has one.
        pytest.importorskip("astropy")
        from astropy.time import Time

        dates = [
            (1970, 1, 1.0, "1970-01-01T00:00:00"),
            (2000, 2, 29.5, "2000-02-29T12:00:00"),
            (2023, 3, 1.25, "2023-03-01T06:00:00"),
            (2024, 12, 31.75, "2024-12-31T18:00:00"),
        ]
        for year, month, day, timestamp in dates:
            expected = Time(timestamp, format="isot", scale="utc").jd
            assert float(conv.gregorian_to_jd(year, month, day)) == pytest.approx(
                expected, rel=0.0, abs=1e-9
            )
