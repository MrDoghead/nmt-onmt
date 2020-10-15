# coding=utf-8

from datetime import date, timedelta, datetime
import time


def get_yesterday(time_format='%Y%m%d'):
    return (date.today() - timedelta(1)).strftime(time_format)


def get_current(time_format='%Y%m%d'):
    local_time = time.localtime()
    time_string = time.strftime(time_format, local_time)
    return time_string


def get_date(diff_days=0, diff_hours=0, day_format='%Y%m%d', current_date=None):
    date={}
    if current_date is None:
        timestamp = time.time() - (diff_days * 24 + diff_hours) * 3600
        date['day'], date['hour'] = time.strftime(day_format + ' %H:%M', time.localtime(timestamp)).split()
        return date

    day = current_date.get('day', None)
    hour = current_date.get('hour', None)
    if hour is None:
        timestamp = time.mktime(time.strptime(day, day_format)) - diff_days * 24 * 3600
        date['day'] = time.strftime(day_format, time.localtime(timestamp))
        return date
    else:
        timestamp = time.mktime(time.strptime('%s %s' % (day, hour), day_format + ' %H:%M')) - \
            (diff_days * 24 + diff_hours) * 3600
        date['day'], date['hour'] = time.strftime(day_format + ' %H:%M', time.localtime(timestamp)).split()
        return date


def get_today(time_format='%Y%m%d'):
    return(date.today()).strftime(time_format)


def get_last_half_hour(time_format='%H-%M-%S'):
    localtime = time.localtime()
    minute = localtime.tm_min

    if minute > 30:
        minute = 30
    else:
        minute = 0

    last_hour_time = str(localtime.tm_year) + convert_digit_to_str(localtime.tm_mon) + convert_digit_to_str(localtime.tm_mday) + ' '\
        + convert_digit_to_str(localtime.tm_hour) + '-' + convert_digit_to_str(minute) + '-' + '00'
    timestamp = time.mktime(time.strptime(last_hour_time, '%Y%m%d %H-%M-%S'))

    return time.strftime(time_format, time.localtime(timestamp))


def convert_digit_to_str(digit):
    if digit < 10:
        return str(0) + str(digit)
    return str(digit)


def get_before_last_half_hour(time_format='%H-%M-%S'):
    localtime = time.localtime()
    minute = localtime.tm_min
    if minute > 30:
        minute = 30
    else:
        minute = 0

    last_hour_time = str(localtime.tm_year) + convert_digit_to_str(localtime.tm_mon) + convert_digit_to_str(localtime.tm_mday) + ' '\
        + convert_digit_to_str(localtime.tm_hour) + '-' + convert_digit_to_str(minute) + '-' + '00'
    timestamp = time.mktime(time.strptime(last_hour_time, '%Y%m%d %H-%M-%S')) - (30 * 60)
    return time.strftime(time_format, time.localtime(timestamp))


def get_week_before(day_time, time_format='%Y%m%d'):
    timestamp = time.mktime(time.strptime(day_time, time_format))
    timestamp -= 7*24*3600
    return time.strftime(time_format, time.localtime(timestamp))


def get_day_before(day_time, num_day, time_format='%Y%m%d'):
    timestamp = time.mktime(time.strptime(day_time, time_format))
    timestamp -= num_day*24*3600
    return time.strftime(time_format, time.localtime(timestamp))


def get_minute_before(time_str, minute, time_format='%H-%M'):
    timestamp = time.mktime(time.strptime(time_str, time_format))
    timestamp -= minute * 60
    return time.strftime(time_format, time.localtime(timestamp))


def convert_time(time_str, input_format, out_format):
    try:
        timestamp = time.mktime(time.strptime(time_str, input_format))
    except OverflowError:
        print("time_str is: ", time_str, "out of range")
        return None
    return time.strftime(out_format, time.localtime(timestamp))


def get_diff_days(time_str1, time_str2, time_format="%Y%m%d"):
    """
    用来计算不同时间相差了几天，注意，这里的时间必须得精度覆盖到天， 精确到小于天的单位会忽略掉
    :param time_str1: 时间1
    :param time_str2: 时间2
    :param time_format: 时间的格式
    :return: 天数1比天数2晚几天
    """

    # timestamp1 = int(time.mktime(time.strptime(time_str1, time_format)) / (3600 * 24))
    # timestamp2 = int(time.mktime(time.strptime(time_str2, time_format)) / (3600 * 24))
    # res_day = timestamp1 - timestamp2
    time1 = datetime.strptime(time_str1, time_format).replace(hour=0, minute=0, second=0, microsecond=0)
    time2 = datetime.strptime(time_str2, time_format).replace(hour=0, minute=0, second=0, microsecond=0)
    return (time1 - time2).days


def from_unix_time(timestamp, time_format='%Y%m%d'):
    return time.strftime(time_format, time.localtime(timestamp))


def get_timestamp(time_str, time_format="%Y%m%d"):
    return time.mktime(time.strptime(time_str, time_format))


def get_today_before(days, time_format='%Y%m%d'):
    """
    获取几天之前的时间
    :param days: 天数， 可以为-数
    :param time_format: 时间的输出格式
    :return:
    """
    return (date.today() - timedelta(days)).strftime(time_format)


if __name__ == '__main__':
    print(get_yesterday())
    print(get_diff_days("20190801", "20190725", "%Y%m%d"))
    print(get_timestamp("2019-08-12 16:45:20", time_format="%Y-%m-%d %H:%M:%S"))
    print(get_timestamp("2019-08-12 16:45:21", time_format="%Y-%m-%d %H:%M:%S"))
    today = get_today(time_format="%Y-%m-%d %H:%M:%S")
    print(today)
    print(get_diff_days(today, "2019-09-01 10:04:14", time_format="%Y-%m-%d %H:%M:%S"))
    print(get_today_before(0))
    print(get_today_before(1))
    print(get_today_before(-1, time_format="%Y-%m-%d %H:%M:%S"))

