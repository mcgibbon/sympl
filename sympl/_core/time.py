from datetime import datetime as real_datetime, timedelta
from .exceptions import DependencyError
try:
    import netcdftime as nt
except ImportError:
    nt = None


def datetime(
        year, month, day, hour=0, minute=0, second=0, microsecond=0,
        tzinfo=None, calendar='proleptic_gregorian'):
    """
    Retrieves a datetime-like object with the requested calendar. Calendar types
    other than proleptic_gregorian require the netcdftime module to be
    installed.

    Parameters
    ----------
    year : int,
    month  : int,
    day  : int,
    hour  : int, optional
    minute  : int, optional
    second : int, optional
    microsecond : int, optional
    tzinfo  : datetime.tzinfo, optional
        A timezone informaton class, such as from pytz. Can only be used with
        'proleptic_gregorian' calendar, as netcdftime does not support
        timezones.
    calendar : string, optional
        Should be one of 'proleptic_gregorian', 'no_leap', '365_day',
        'all_leap', '366_day', '360_day', 'julian', or 'gregorian'. Default
        is 'proleptic_gregorian', which returns a normal Python datetime.
        Other options require the netcdftime module to be installed.

    Returns
    -------
    datetime : datetime-like
        The requested datetime. May be a Python datetime, or one of the
        datetime-like types in netcdftime.
    """
    kwargs = {
        'year': year, 'month': month, 'day': day, 'hour': hour,
        'minute': minute, 'second': second, 'microsecond': microsecond
    }
    if calendar.lower() == 'proleptic_gregorian':
        return real_datetime(tzinfo=tzinfo, **kwargs)
    elif tzinfo is not None:
        raise ValueError('netcdftime does not support timezone-aware datetimes')
    elif nt is None:
        raise DependencyError(
            "Calendars other than 'proleptic_gregorian' require the netcdftime "
            "package, which is not installed.")
    elif calendar.lower() in ('all_leap', '366_day'):
        return nt.DatetimeAllLeap(**kwargs)
    elif calendar.lower() in ('no_leap', 'noleap', '365_day'):
        return nt.DatetimeNoLeap(**kwargs)
    elif calendar.lower() == '360_day':
        return nt.Datetime360Day(**kwargs)
    elif calendar.lower() == 'julian':
        return nt.DatetimeJulian(**kwargs)
    elif calendar.lower() == 'gregorian':
        return nt.DatetimeGregorian(**kwargs)


__all__ = (datetime, timedelta)
