"""This module contains the classes
 in an ectt file that is courses and rooms"""


class Header:
    """Describes the meta data of the university timetable problems"""

    def __init__(self, name, num_courses, num_rooms, num_days,
                 num_period_per_day,
                 num_curricula, min_daily_lect, max_daily_lect,
                 num_unavail_constr, num_room_const):
        self.name = name
        self.num_courses = int(num_courses)
        self.num_rooms = int(num_rooms)
        self.num_days = int(num_days)
        self.num_period_per_day = int(num_period_per_day)
        self.num_curricula = int(num_curricula)
        self.min_daily_lect = int(min_daily_lect)
        self.max_daily_lect = int(max_daily_lect)
        self.num_unavail_constr = int(num_unavail_constr)
        self.num_room_const = int(num_room_const)

    def getmaximumperiod(self):
        """Get maximum possible period value in timetable schedule
        period starts of 0"""
        return (self.num_days * self.num_period_per_day) - 1

    def getnumperiods(self):
        """Get the total number of periods in ectt instance"""
        return self.num_days * self.num_period_per_day

    def getday(self, period):
        """Get the day of a given period day starts from 0"""

        # // gets the integer (whole) part of division
        return period // self.num_period_per_day

    def getperiodinday(self, period):
        """ Get the period in day (slot) given  period value
        day period starts at 0"""
        return period % self.num_period_per_day

    def getperiod(self, period_in_day, day):
        """Get period in range [0, getmaximumperiod]
        given day and period in day"""
        return (self.num_period_per_day * day) + period_in_day

    def __str__(self):
        " Returns a dictionary of object"
        return str(self.__dict__)


class Course:
    """ Decribes a course object in an ectt instance"""

    def __init__(self, coursecode, lecturer, lect_per_week,
                 min_working_days, num_studs, num_double_periods):
        self.coursecode = coursecode
        self.lecturer = lecturer
        self.lect_per_week = int(lect_per_week)
        self.min_working_days = int(min_working_days)
        self.num_stud = int(num_studs)
        self.num_double_periods = int(num_double_periods)

    def __str__(self):
        " Returns a dictionary of object"
        return str(self.__dict__)


class Room:
    """ Object for venue a lecture will hold"""

    def __init__(self, roomid, size, site):
        self.roomid = roomid
        self.size = int(size)
        self.site = site

    def __str__(self):
        " Returns a dictionary of object"
        return str(self.__dict__)


class Curricula:
    """ Object for list of courses in curricla in ecct data"""

    def __init__(self, currid, num_courses, list_courses):
        self.currid = currid
        self.num_courses = int(num_courses)
        self.list_courses = list_courses

    def __str__(self):
        " Returns a dictionary of object"
        return str(self.__dict__)


class UnavailConstr:
    """ period when a course lecture cannot hold"""

    def __init__(self, coursecode, day, period_in_day, num_period_per_day):
        self.coursecode = coursecode
        self.period = (int(num_period_per_day) * int(day)) + int(period_in_day)

    def __str__(self):
        " Returns a dictionary of object"
        return str(self.__dict__)


class RoomConstr:
    """ Venues that courses must hold in"""

    def __init__(self, coursecode, roomid):
        self.coursecode = coursecode
        self.roomid = roomid

    def __str__(self):
        " Returns a dictionary of object"
        return str(self.__dict__)


class Event:
    """A course room day period combination
    List of events make a timetable solution"""

    def __init__(self, coursecode, roomid, day, period_in_day, period):
        self.coursecode = coursecode
        self.roomid = roomid
        self.day = day
        self.period_in_day = period_in_day
        self.period = period

    def __str__(self):
        " Returns a dictionary of object"
        return str(self.__dict__)

    def filewrite(self):
        """Format of event for writing to solution file"""
        return str(self.coursecode) + ' ' + str(self.roomid) + ' '\
            + str(self.day) + ' ' + str(self.period_in_day) + '\n'


class RoomPeriod:
    """A venue and time slot combination """

    def __init__(self, roomid, period):
        self.roomid = roomid
        # period starts at zero to max (day * period in day) - 1iu
        self.period = int(period)

    def __str__(self):
        " Returns a dictionary of object"
        return str(self.__dict__)

    def __eq__(self, other):
        "This method overides the equal comparator so == can be used"
        return self.__dict__ == other.__dict__
