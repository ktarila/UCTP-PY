"""This module reads a ectt file. And contains methods for
hard and soft constraint evaluations"""
# import os
from os.path import abspath, join
import copy
import random
import itertools
import operator
from collections import Counter
import ectt_classes


# ----------------Read ECTT File---------------------------------------------
def readecttfile(fullpath):
    """Read ectt file and return objects"""
    # fullpath = join(abspath('../InputData/Test_ectt'), filename)
    # fullpath = join(abspath('../InputData/ITC-2007_ectt'), filename)
    location = 'HEADER'
    headtext = []
    coursestext = []
    roomstext = []
    curriculatext = []
    unavailconstrtext = []
    roomconstrtext = []
    with open(fullpath) as ectt_file:
        for line in ectt_file:
            if len(line) > 1:
                if line == 'COURSES:\n':
                    location = 'COURSES'
                if line == 'ROOMS:\n':
                    location = 'ROOMS'
                if line == 'CURRICULA:\n':
                    location = 'CURRICULA'
                if line == 'UNAVAILABILITY_CONSTRAINTS:\n':
                    location = 'UNAVAILABILITY_CONSTRAINTS'
                if line == 'ROOM_CONSTRAINTS:\n':
                    location = 'ROOM_CONSTRAINTS'
                if line == 'END.\n':
                    location = 'END'
                    # print("finished reading input file")

                if location == 'HEADER':
                    headtext.append(line.replace('\r', '').replace('\n', ''))
                if (location == 'COURSES') & ('COURSES' not in line):
                    coursestext.append(line.replace(
                        '\r', '').replace('\n', ''))
                if (location == 'ROOMS') & ('ROOMS' not in line):
                    roomstext.append(line.replace(
                        '\r', '').replace('\n', ''))
                if (location == 'CURRICULA') & ('CURRICULA' not in line):
                    curriculatext.append(line.replace(
                        '\r', '').replace('\n', ''))
                if ((location == 'UNAVAILABILITY_CONSTRAINTS') &
                        ('UNAVAILABILITY_CONSTRAINTS' not in line)):
                    unavailconstrtext.append(line.replace(
                        '\r', '').replace('\n', ''))
                if ((location == 'ROOM_CONSTRAINTS') &
                        ('ROOM_CONSTRAINTS' not in line)):
                    roomconstrtext.append(line.replace(
                        '\r', '').replace('\n', ''))

    # print(headtext, "Header")
    # print(coursestext, "Courses")
    # print(roomstext, "Rooms")
    # print(curriculatext, "Curricula")
    # print(unavailconstrtext, "Unavailable constraints")
    # print(roomconstrtext, "Room Constraints")
    # return (headtext, coursestext, roomstext, curriculatext,
    #         unavailconstrtext, roomconstrtext)
    header = createheader(headtext)
    return (header, createcourselist(coursestext),
            createroomlist(roomstext), createcurriculalist(curriculatext),
            createunavailconstrlist(unavailconstrtext,
                                    header.num_period_per_day),
            createroomconstrlist(roomconstrtext))


def createheader(htext):
    """Create header object for ectt instance"""
    return ectt_classes.Header(htext[0].split()[1],  # Instance name
                               htext[1].split()[1],  # num courses
                               htext[2].split()[1],  # num rooms
                               htext[3].split()[1],  # num days in week
                               htext[4].split()[1],  # num periods in day
                               htext[5].split()[1],  # num curricula
                               htext[6].split()[1],  # min daily lect
                               htext[6].split()[2],  # max daily lect
                               htext[7].split()[1],  # num unavail constr
                               htext[8].split()[1])  # num room constr


def createcourselist(ctext):
    """Create list of courses in ectt instance"""
    courselist = []
    courselecturerlist = []
    course_count = {}
    course_size = {}
    course_minwday = {}
    for course in ctext:
        courselist.append((ectt_classes.Course(
            course.split()[0],  # coursecode,
            course.split()[1],  # lecturer,
            course.split()[2],  # number lect_per_week
            course.split()[3],  # min days lecture spread in a week,
            course.split()[4],  # number of students,
            course.split()[5])))  # number of double_periods
        courselecturer = {"coursecode": course.split()[0],
                          "lecturer": course.split()[1]}
        courselecturerlist.append(courselecturer)  # lecturer name
        course_count[course.split()[0]] = course.split()[2]
        course_size[course.split()[0]] = course.split()[4]
        course_minwday[course.split()[0]] = course.split()[3]
    return (courselist, courselecturerlist,
            course_count, course_size, course_minwday)


def createroomlist(rtext):
    """Create list of rooms in ectt instance"""
    roomlist = []
    room_size = {}
    room_site = {}
    for room in rtext:
        roomlist.append((ectt_classes.Room(
            room.split()[0],  # room id
            room.split()[1],  # room size
            room.split()[2])))  # site (room location)
        room_size[room.split()[0]] = room.split()[1]
        room_site[room.split()[0]] = room.split()[2]
    return (roomlist, room_size, room_site)


def createcurriculalist(currtext):
    """Create a list of curricula in ectt instance"""
    currlist = []
    curriculatakingcourse = {}
    for curriculum in currtext:
        curriculu = curriculum.split()
        array_courses = []
        for i in range(2, len(curriculu), 1):
            array_courses.append(curriculu[i])
            # append course in ith location of curricula location of input file
            key = curriculu[i]
            if key in curriculatakingcourse:
                curriculatakingcourse[key].append(curriculu[0])
            else:
                curriculatakingcourse[key] = [curriculu[0]]

        currlist.append((ectt_classes.Curricula(
            curriculu[0],
            curriculu[1],
            array_courses)))
    return (currlist, curriculatakingcourse)


def createunavailconstrlist(unavailtext, num_slots_in_day):
    """Create list of unavailable times for course"""
    unavaillist = []
    course_unavail_periods = {}
    for unavailslot in unavailtext:
        # unavaillist.append(ectt_classes.UnavailConstr(
        #     unavailslot.split()[0],  # couse code
        #     unavailslot.split()[1],  # day
        #     unavailslot.split()[2],  # period in day
        #     num_slots_in_day))  # number of periods in a day
        period = (int(
            num_slots_in_day) * int(
                unavailslot.split()[1])) + int(unavailslot.split()[2])
        unavail_constr = {"coursecode": unavailslot.split()[0],
                          "period": period}
        unavaillist.append(unavail_constr)
        key = unavailslot.split()[0]
        if key in course_unavail_periods:
            course_unavail_periods[key].append(period)
        else:
            course_unavail_periods[key] = [period]
    return unavaillist, course_unavail_periods


def createroomconstrlist(roomconstrtext):
    """Create a list of room constraits
       that is rooms that must hold lectures for particular
       course"""
    roomconstrlist = []
    for roomconstr in roomconstrtext:
        roomconstrlist.append(ectt_classes.RoomConstr(
            roomconstr.split()[0],  # course code
            roomconstr.split()[1]))  # room id
    return roomconstrlist


# ----------------Create Timetable solution-----------------------------------
def createroomperiodlist(roomlist, numperiods):
    """Create a room period list for ectt instance"""
    roomperiodlist = []
    for venue in roomlist:
        for per in range(0, numperiods):  # range doesn't inclued end
            roomperiodlist.append(
                ectt_classes.RoomPeriod(venue.roomid, per))
    return roomperiodlist


def createlecturelist(courses):
    """create list of courses that should be scheduled"""
    lecturelist = []
    for csc in courses:
        for _ in range(csc.lect_per_week):
            lecturelist.append(csc.coursecode)
    return lecturelist


def createrandomsolution(numlectures, numroomperiods):
    """create a random solution as list"""
    # random range without repetition of roomperiod
    # return random.sample(range(numroomperiods), numlectures)
    # random in range with repetition
    return [
        random.randint(0, (numroomperiods - 1)) for r in range(numlectures)]


def scheduletosolution(timetable, rplist, lecturelist):
    """Revert a schedule to a solution of list of integers"""
    solution = []
    # courses = inputdata[1][0]
    # print(len(timetable))
    schedule = list(timetable)
    for event in lecturelist:
        # print(event, len(schedule))
        found = False
        index = 0
        while found is False and index < len(schedule):
            # print(index, schedule[index].coursecode)
            event_tuple = schedule[index]
            if event_tuple.coursecode == event:
                # print("\t", index, event_tuple, event, "found here")
                # print(schedule[index])
                temp_rp = ectt_classes.RoomPeriod(
                    event_tuple.roomid, event_tuple.period)
                rp_index = rplist.index(temp_rp)
                # print(rplist[rp_index])
                # print(temp_rp)
                del schedule[index]
                solution.append(rp_index)
                found = True
            index += 1
        if found is False:
            # print(event, "No assigned roomperiod for event")
            # assign a rp to event -- even though it may
            # cause a violation
            solution.append(random.randint(0, (len(rplist) - 1)))
        # else:
        #     print(event, "Does exist")
        # print("\n")

    return solution


def createschedule(randsolution, rmplist, lectlist, header):
    """Create a timetable schedule, list of events"""
    eventlist = []
    cp_duplic_check = []
    for i, _ in enumerate(lectlist):  # enumerate better _ for other val
        rmpindex = randsolution[i]
        temp = (lectlist[i], rmplist[rmpindex].period)
        if temp not in cp_duplic_check:
            cp_duplic_check += [temp]
            eventlist.append(ectt_classes.Event(
                lectlist[i],  # course code
                rmplist[rmpindex].roomid,  # roomid
                header.getday(rmplist[rmpindex].period),  # day
                header.getperiodinday(rmplist[rmpindex].period),  # per in day
                rmplist[rmpindex].period))  # period in 0 to max possible per
        # else:
        #     print("Warning !!! Course period duplicate: ", temp)
    return eventlist


def createconstructsequece(lectlist):
    """Create sequece for constructing timetable"""
    event_list = list(lectlist)
    ind = random.sample(range(len(event_list)), len(event_list))
    return ind


def constructschedule(individual, rmplist, lectlist, data):
    """Create a timetable schedule, list containing assign order"""

    inputdata = copy.copy(data)
    header = inputdata[0]
    avail_rp = list(rmplist)
    cur_period = {}
    lecturer_period = {}
    for i in inputdata[3][0]:
        cur_period[i.currid] = []
    # print(len(avail_rp))
    for i in inputdata[1][0]:
        lecturer_period[i.lecturer] = []
    # print(cur_period)
    # print("\n", lecturer_period)

    eventlist = []
    cp_duplic_check = []

    for i in individual:
        # print(i, len(cfg.LECTURELIST))
        rmpindex, avail_rp = getrpindex(
            lectlist[i],
            avail_rp, cur_period, lecturer_period, inputdata, rmplist)
        # print(len(avail_rp), "Length of available period")
        temp = (lectlist[i], rmplist[rmpindex].period)
        if temp not in cp_duplic_check:
            cp_duplic_check += [temp]
            eventlist.append(ectt_classes.Event(
                lectlist[i],  # course code
                rmplist[rmpindex].roomid,  # roomid
                header.getday(rmplist[rmpindex].period),  # day
                header.getperiodinday(rmplist[rmpindex].period),  # per in day
                rmplist[rmpindex].period))  # period in 0 to max possible per
        # else:
        #     print("Warning !!! Course period duplicate: ", temp)
    return eventlist


def getrpindex(event, avail_rp, cur_period, lecturer_period, inputdata, rplst):
    """Get a possible rt for event"""
    # print(inputdata[4][1])
    if event in inputdata[4][1]:
        unavail = list(inputdata[4][1][event])
    else:
        unavail = []

    # add periods taken for current curricula to unavailable
    csc_curr = set(inputdata[3][1][event])
    for cur, periods in cur_period.items():
        if cur in csc_curr:
            unavail.extend(periods)

    # csc_lecturer = inputdata[1][0].lecturer
    # add periods taken for current lecturer
    lecturer = None
    course = [course for course in inputdata[1][0]
              if course.coursecode == event]
    if len(course) > 0:
        lecturer = course[0].lecturer
        unavail.extend(lecturer_period[lecturer])

    possible_room = [room for room in avail_rp if room.period not in unavail]

    # random.seed(64)
    # not selecting random so resulting schedule for individual
    # is consistent
    if len(possible_room) > 0:
        # assign_roomperiod = random.choice(possible_room)
        assign_roomperiod = possible_room[0]
    else:
        # available will cause unavailability constraint
        # assign_roomperiod = random.choice(avail_rp)
        assign_roomperiod = avail_rp[0]
        # print("assigning unavailable or curricula clash")

    try:
        rp_index = rplst.index(assign_roomperiod)
    except ValueError:
        rp_index = random.randint(0, len(rplst) - 1)

    # remove taken rp from avail_rp
    for index, val in enumerate(avail_rp):
        if (val.period == rplst[rp_index].period and
                val.roomid == rplst[rp_index].roomid):
            # delete specified index
            del avail_rp[index]

    # add curriculum to period
    for item in csc_curr:
        cur_period[item].append(rplst[rp_index].period)

    # add lecturer to period
    lecturer_period[lecturer].append(rplst[rp_index].period)

    return rp_index, avail_rp


def writescheduletofile(eventlist, filename):
    """Write a timetable solution to a file"""
    fullpath = join(abspath('../Solution'), filename)
    file = open(fullpath, "w")
    for event in eventlist:
        # print(event.coursecode, event.day, event.period_in_day)
        file.write(event.filewrite())
    file.close()


# ----------------Hard constraint evaluation-----------------------------------
def getgroupsforevaluation(eventlist):
    """Get the groups to compute evaluation"""
    period_course = {}  # grp by period for curr & lect clash
    course_period_list = []
    period_room_list = {}
    lect_in_sch = []
    for row in eventlist:
        key_pc = row.period
        key_room = row.period
        lect_in_sch.append(row.coursecode)
        if key_pc in period_course:
            period_course[key_pc].append(row.coursecode)
        else:
            period_course[key_pc] = [row.coursecode]
        if key_room in period_room_list:
            period_room_list[key_room].append((row.coursecode, row.roomid))
        else:
            period_room_list[key_room] = [(row.coursecode, row.roomid)]
        # get course period list for unavail check
        cp_list = {"coursecode": row.coursecode,
                   "period": row.period}
        course_period_list.append(cp_list)
    return (period_course,
            course_period_list, period_room_list, lect_in_sch)


def getunassignedviolations(lectures_in_sch, course_lect_per_week):
    """Get the number of courses unassigned

    lectures_in_sch: dictionary count of number of weekly lectures for
                     each course in schedule
    course_lect_per_week: dictionary count of number of weekly required
                        lectures as per input

    Voilation on each course weighs as one on ITV validator even though
    Could affect genetic algorithm so voilation weight  may be changed
    """
    sum_violations = 0
    for key, _ in course_lect_per_week.items():
        difference = int(course_lect_per_week[key]) - int(lectures_in_sch[key])
        sum_violations += difference
    return sum_violations


def getstudlectclashviolations(
        periodkey_cscval_dict, cur_taking_csc_dict, courselecturerlist):
    """Get number of student clash violations
    that is same curricula in a period and lecturer clash violations
    that is lecturer assigned to teach two periods in same day"""
    sum_violations = 0
    for key in periodkey_cscval_dict:
        period_courses = periodkey_cscval_dict[key]
        courses_clash = []
        # print(key, period_courses)
        # get course pairs
        combinatns = (list(
            itertools.combinations(range(len(period_courses)), 2)))
        for first_index, second_index in combinatns:
            course1 = period_courses[first_index]
            course2 = period_courses[second_index]
            # Get curriculum clash
            cur_clash = list(set(cur_taking_csc_dict[course1]) & set(
                cur_taking_csc_dict[course2]))
            # check if there is lecturer clash in pair
            lect_clash = False
            lecturer1 = next((item for item in courselecturerlist if item[
                "coursecode"] == course1))
            lecturer2 = next((item for item in courselecturerlist if item[
                "coursecode"] == course2))
            if lecturer1['lecturer'] == lecturer2['lecturer']:
                # print("lecturer clash", lecturer1, lecturer2)
                lect_clash = True
            if len(cur_clash) > 0 or lect_clash:
                courses_clash += [(course1, course2)]
        # add violations to global
        sum_violations += len(courses_clash)
    return sum_violations


def getunavailperiodviolations(all_cscper_list, unavail_cscper_list):
    """Get number of courses in unavailable periods"""
    # intersect two lists ... size of output is num violations
    voi = [i for i in unavail_cscper_list
           for j in all_cscper_list
           if i['coursecode'] == j['coursecode']and i['period'] == j['period']]
    # print(len(voi))
    # return len(voi)
    # remove duplicate from course -- duplicates are ignored in validator
    return len(set([tuple(d.items()) for d in voi]))


def getroomclashviolations(room_per_list):
    """Get the room clashes for each period"""
    sum_violations = 0
    for item in room_per_list:
        newroomlist = (list(grouptuplebyitem(room_per_list[item])))
        per_rm_viol = len(newroomlist) - len(set(newroomlist))
        # print(item, newroomlist, per_rm_viol)
        sum_violations += per_rm_viol
    return sum_violations


def grouptuplebyitem(tuple_list):
    """Group tuple by first attribute"""
    iterat = itertools.groupby(tuple_list, operator.itemgetter(0))
    for _, subiter in iterat:
        # yield key, [item[1] for item in subiter]
        # return first match
        value = [item[1] for item in subiter]
        # return first item as cc validator ignores others (duplicate csc)
        yield value[0]


def gethardconstraintsviolations(schedule, inputdata):
    """Get number of hard const violations"""
    (periodcourse_dict,
     cp_list,
     period_venues,
     lect_scheduled) = getgroupsforevaluation(schedule)

    count_in_schedule = dict(Counter(lect_scheduled))

    # UNASSIGNED VIOLATIONS
    unassignedviolations = getunassignedviolations(
        count_in_schedule, inputdata[1][2])

    # CURRICULUM VIOLATIONS
    lectstudclashviolations = getstudlectclashviolations(
        periodcourse_dict, inputdata[3][1], inputdata[1][1])

    # UNAVAILABLE PERIOD VIOLATIONS
    unavailviolations = getunavailperiodviolations(cp_list, inputdata[4][0])

    # ROOM CLASH VIOLATIONS
    venueviolations = getroomclashviolations(period_venues)

    # print("\tunassigned violations", unassignedviolations)
    # print("\tlecturer or stud violations", lectstudclashviolations)
    # print("\tviolations of availability", unavailviolations)
    # print("\tviolations of room occupation", venueviolations)
    return int(unassignedviolations +
               lectstudclashviolations + unavailviolations + venueviolations)


# ----------------Soft Constraints Evaluation----------------------------------
def getgroupforsofteval(timetable):
    """Get groups to evaluate softconstraints"""
    course_room = []
    p_csc = {}
    day_csc = {}
    course_room_dict = {}
    csc_day = {}
    for event in timetable:
        k_pc = event.period
        key_day = event.day
        key_csc = event.coursecode
        c_room = {"coursecode": event.coursecode, "roomid": event.roomid}
        course_room.append(c_room)
        if k_pc in p_csc:
            p_csc[k_pc].append(event.coursecode)
        else:
            p_csc[k_pc] = [event.coursecode]
        if key_day in day_csc:
            day_csc[key_day].append(event.coursecode)
        else:
            day_csc[key_day] = [event.coursecode]
        if key_csc in course_room_dict:
            course_room_dict[key_csc].append(event.roomid)
        else:
            course_room_dict[key_csc] = [event.roomid]
        if key_csc in csc_day:
            csc_day[key_csc].append(event.day)
        else:
            csc_day[key_csc] = [event.day]

    return (course_room, p_csc, course_room_dict, csc_day)


def getroomsizeviolations(c_room, r_size, c_size):
    """Get the number of room size violations"""
    sum_violations = 0
    for item in c_room:
        if int(c_size[item['coursecode']]) > int(r_size[item['roomid']]):
            difference = int(c_size[item['coursecode']]) - int(
                r_size[item['roomid']])
            # print("Room size: ", r_size[item['roomid']],
            #       "course size: ", c_size[item['coursecode']])
            sum_violations += difference
    return sum_violations


def getperiodcurricula(csc_curr, period_csc):
    """Get period curricula"""
    # print("\n\n", period_csc)
    # print("\n\n", csc_curr)
    period_curr_dict = {}
    for key, value in period_csc.items():
        # print(value)
        period_curr = [csc_curr[x] for x in value]
        # print("\t", period_curr)
        # flat_pr = set([item for sublist in period_curr for item in sublist])
        flat_pr = [item for sublist in period_curr for item in sublist]
        period_curr_dict[key] = flat_pr
    # print("\n\n\n", period_curr_dict)
    return period_curr_dict


def getisolatedlectureviolations(csc_curr, period_csc, header):
    """curriculum that doesn't have lectures adjacent to each other"""
    sum_violations = 0
    # print(header)

    per_curr = getperiodcurricula(csc_curr, period_csc)
    for day in range(0, header.num_days):
        for per_in_day in range(0, header.num_period_per_day):
            # print("Day:", day, end=" ")
            # print("\tPeriod in day: ", per_in_day, end=" ")
            adjacent_periods = []
            adj_per = []
            period = header.getperiod(per_in_day, day)
            # print("Period in week: ", period)

            # get adjacent periods
            if per_in_day != 0:
                adj_per.append(per_in_day - 1)
                adjacent_periods.append(
                    header.getperiod((per_in_day - 1), day))
            if per_in_day != (header.num_period_per_day - 1):
                adj_per.append(per_in_day + 1)
                adjacent_periods.append(
                    header.getperiod((per_in_day + 1), day))
            # print(adjacent_periods, adj_per)

            # Get curricula in adjacent periods
            adj_union = set()
            for adj in adjacent_periods:
                # print("\t", header.getperiodinday(adj), adj, end=" ")
                if adj in per_curr:
                    # print(per_curr[adj])
                    adj_union = adj_union.union(set(per_curr[adj]))
            # print("\tadjacent curricula: ", adj_union)

            # get curricula in current period
            single_per_curr = []
            if period in per_curr:
                single_per_curr = per_curr[period]
            # print("\tCurrent period curricula", single_per_curr)

            # subtract current period curriculum frm adj curr to get isolated
            isolated = [val for val in single_per_curr if val not in adj_union]
            # print("\tisolated: ", isolated)

            # add period violation to global
            sum_violations += len(isolated)
            # print("\n")
            # course_cur =
    return sum_violations * 2


def getroombalanceviolations(csc_room_list):
    """Get lectures not in same room violations"""
    sum_violations = 0
    for _, item in csc_room_list.items():
        # print(key, item, end=" ")
        # number of diff rooms used -1 to get num extra rooms
        csc_vio = len(set(item)) - 1
        # print(csc_vio)
        sum_violations += csc_vio
    return sum_violations


def getmwdviolation(course_days, csc_mwd):
    """Minimum working days violation"""
    sum_violations = 0
    # print(course_days)
    for key, item in course_days.items():
        # print(key, set(item), csc_mwd[key], end=" ")
        # min work day violation
        minwd_vio = 0
        if len(set(item)) < int(csc_mwd[key]):
            minwd_vio = int(csc_mwd[key]) - len(set(item))
            # print(key, set(item), csc_mwd[key], "Fewer days: ", minwd_vio)
        # print("Fewer days: ", minwd_vio)
        sum_violations += minwd_vio
    return sum_violations * 5


def getsoftconstraintviolations(timetable, inputdata):
    """Get the number of soft constraint violations in timetable"""

    (c_room,
     period_csc,
     c_room_dict,
     course_days) = getgroupforsofteval(timetable)

    # Get room size violations
    room_size_vio = getroomsizeviolations(
        c_room, inputdata[2][1], inputdata[1][3])

    isolated_cur_vio = getisolatedlectureviolations(
        inputdata[3][1], period_csc, inputdata[0])

    room_bal_vio = getroombalanceviolations(c_room_dict)

    min_work_days_vio = getmwdviolation(course_days, inputdata[1][4])

    # print("\tRoom size violations: ", room_size_vio)
    # print("\tMin working days violations: ", min_work_days_vio)
    # print("\tIsolated curriculum violations: ", isolated_cur_vio)
    # print("\tRoom stability violations: ", room_bal_vio)
    return int(room_size_vio +
               isolated_cur_vio + room_bal_vio + min_work_days_vio)
