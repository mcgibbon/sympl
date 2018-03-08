from datetime import datetime as real_datetime
from sympl import datetime, timedelta
import unittest
import pytz
try:
    import netcdftime as nt
except ImportError:
    nt = None

netcdftime_not_installed = 'netcdftime module is not installed'


class DatetimeBase(object):

    dt_class = None
    calendar = None

    def min_args_dt(self):
        return datetime(2005, 10, 12, calendar=self.calendar)

    def max_args_dt(self):
        return datetime(202, 5, 6, 7, 8, 9, 10, calendar=self.calendar)

    def testMinArgDatetimeIsCorrectClass(self):
        assert isinstance(self.min_args_dt(), self.dt_class)

    def testMaxArgDatetimeIsCorrectClass(self):
        assert isinstance(self.max_args_dt(), self.dt_class)

    def testMinArgDatetimeHasCorrectValues(self):
        min_args_dt = self.min_args_dt()
        assert min_args_dt.year == 2005
        assert min_args_dt.month == 10
        assert min_args_dt.day == 12
        assert min_args_dt.hour == 0
        assert min_args_dt.minute == 0
        assert min_args_dt.second == 0
        assert min_args_dt.microsecond == 0

    def testMaxArgDatetimeHasCorrectValues(self):
        max_args_dt = self.max_args_dt()
        assert max_args_dt.year == 202
        assert max_args_dt.month == 5
        assert max_args_dt.day == 6
        assert max_args_dt.hour == 7
        assert max_args_dt.minute == 8
        assert max_args_dt.second == 9
        assert max_args_dt.microsecond == 10



class ProlepticGregorianTests(unittest.TestCase, DatetimeBase):

    dt_class = real_datetime
    calendar = 'proleptic_gregorian'

    def tz_dt(self):
        return datetime(2003, 9, 30, tzinfo=pytz.timezone('US/Eastern'), calendar=self.calendar)

    def testTimezoneAwareDatetimeIsCorrectClass(self):
        assert isinstance(self.tz_dt(), self.dt_class)

    def testTimezoneAwareDatetimeHasCorrectValues(self):
        tz_dt = self.tz_dt()
        assert tz_dt.year == 2003
        assert tz_dt.month == 9
        assert tz_dt.day == 30
        assert tz_dt.hour == 0
        assert tz_dt.minute == 0
        assert tz_dt.second == 0
        assert tz_dt.microsecond == 0


@unittest.skipIf(nt is None, netcdftime_not_installed)
class NoLeapTests(unittest.TestCase, DatetimeBase):

    calendar = 'no_leap'

    @property
    def dt_class(self):
        return nt.DatetimeNoLeap


@unittest.skipIf(nt is None, netcdftime_not_installed)
class Datetime365DayTests(unittest.TestCase, DatetimeBase):

    calendar = '365_day'

    @property
    def dt_class(self):
        return nt.DatetimeNoLeap

    def test_incrementing_years_using_days(self):
        dt = datetime(1900, 1, 1, calendar=self.calendar)
        dt_1950 = dt + timedelta(days=365 * 50)
        assert dt_1950.year == 1950
        assert dt_1950.month == 1
        assert dt_1950.day == 1
        assert dt_1950.hour == 0

    def test_decrementing_years_using_days(self):
        dt = datetime(1900, 1, 1, calendar=self.calendar)
        dt_1850 = dt - timedelta(days=365 * 50)
        assert dt_1850.year == 1850
        assert dt_1850.month == 1
        assert dt_1850.day == 1
        assert dt_1850.hour == 0


@unittest.skipIf(nt is None, netcdftime_not_installed)
class AllLeapTests(unittest.TestCase, DatetimeBase):
    calendar = 'all_leap'

    @property
    def dt_class(self):
        return nt.DatetimeAllLeap


@unittest.skipIf(nt is None, netcdftime_not_installed)
class Datetime366DayTests(unittest.TestCase, DatetimeBase):
    calendar = '366_day'

    @property
    def dt_class(self):
        return nt.DatetimeAllLeap

    def test_incrementing_years_using_days(self):
        dt = datetime(1900, 1, 1, calendar=self.calendar)
        dt_1950 = dt + timedelta(days=366 * 50)
        assert dt_1950.year == 1950
        assert dt_1950.month == 1
        assert dt_1950.day == 1
        assert dt_1950.hour == 0

    def test_decrementing_years_using_days(self):
        dt = datetime(1900, 1, 1, calendar=self.calendar)
        dt_1850 = dt - timedelta(days=366 * 50)
        assert dt_1850.year == 1850
        assert dt_1850.month == 1
        assert dt_1850.day == 1
        assert dt_1850.hour == 0


@unittest.skipIf(nt is None, netcdftime_not_installed)
class Datetime360DayTests(unittest.TestCase, DatetimeBase):
    calendar = '360_day'

    @property
    def dt_class(self):
        return nt.Datetime360Day

    def test_incrementing_years_using_days(self):
        dt = datetime(1900, 1, 1, calendar=self.calendar)
        dt_1950 = dt + timedelta(days=360 * 50)
        assert dt_1950.year == 1950
        assert dt_1950.month == 1
        assert dt_1950.day == 1
        assert dt_1950.hour == 0

    def test_decrementing_years_using_days(self):
        dt = datetime(1900, 1, 1, calendar=self.calendar)
        dt_1850 = dt - timedelta(days=360 * 50)
        assert dt_1850.year == 1850
        assert dt_1850.month == 1
        assert dt_1850.day == 1
        assert dt_1850.hour == 0


@unittest.skipIf(nt is None, netcdftime_not_installed)
class JulianTests(unittest.TestCase, DatetimeBase):
    calendar = 'julian'

    @property
    def dt_class(self):
        return nt.DatetimeJulian


unittest.skipIf(nt is None, netcdftime_not_installed)
class GregorianTests(unittest.TestCase, DatetimeBase):
    calendar = 'gregorian'

    @property
    def dt_class(self):
        return nt.DatetimeGregorian
