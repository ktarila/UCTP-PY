"""Genetic Operators to create schedule"""
import random
import copy
import readectt
import config as cfg

# print(config.ROOMPERIODLIST)


def find_index(element, list_element):
    """Find index of element in list
    Return none if element not in list"""
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None


# hard constraint operators
def hardconstmove(lecture_index, new_lect_index, individual):
    """Move a lecture to a new position"""
    indi = copy.copy(individual)
    old_value = individual[lecture_index]
    new_value = individual[new_lect_index]

    indi[lecture_index] = new_value
    indi[new_lect_index] = old_value

    return indi


def hardevaluation(individual):
    """hard constraints evaluation function for individual
    in population"""
    # schedule = readectt.createschedule(
    #     individual, cfg.ROOMPERIODLIST, cfg.LECTURELIST, cfg.DATA[0])

    rplist = cfg.ROOMPERIODLIST
    lectlist = cfg.LECTURELIST
    schedule = readectt.constructschedule(
        individual, rplist, lectlist, cfg.DATA)

    hcveval = readectt.gethardconstraintsviolations(schedule, cfg.DATA)
    return hcveval,

    # hcveval *= 1000

    # sceval = readectt.getsoftconstraintviolations(schedule, cfg.DATA)
    # # trailing comma important for single obj deap eval
    # return hcveval*sceval + sceval,


def hardmutate(individual):
    """Hard constraint mutation
    Move to order of assinging a random lect to a random location
    """

    num_lectures = len(cfg.LECTURELIST)

    lect_index = random.randint(0, (num_lectures - 1))
    new_lect_index = random.randint(0, (num_lectures - 1))

    newind = hardconstmove(lect_index, new_lect_index, individual)

    return newind,  # trailing comma important for single obj deap eval


def hardtwopointcrossover(ind1, ind2):
    """Soft Constraint 2 point crossover
    move rt of event in ind1 with ind2"""

    c_slice = len(ind1) // 2  # get cross over point

    individual1 = copy.copy(ind1)
    individual2 = copy.copy(ind2)

    # print(c_slice, "cross over slice")

    # get new offspring from first part of crossover
    for i in range(c_slice):
        # individual1 = simplemovelecture(i, ind2[i], individual1)
        individual1 = hardconstmove(i, ind2[i], individual1)

    # get new offspring from second part of cross over
    for i in range(c_slice, len(ind1)):
        individual2 = hardconstmove(i, ind1[i], individual2)

    return individual1, individual2


# soft constraint operators
def simplemovelecture(lecture_index, new_roomperiod, individual):
    """Move a lecture to a new room period
    if room period is taken then swap room period of two lectures"""
    indi = copy.copy(individual)
    # swap if rand_rp_index is already taken
    swap_rp = find_index(new_roomperiod, indi)
    # print("\t\t\t", swap_rp, "index duplicate")
    if swap_rp is not None:
        old_rp = indi[lecture_index]
        indi[lecture_index] = new_roomperiod
        indi[swap_rp] = old_rp
    else:
        indi[lecture_index] = new_roomperiod

    # print(lecture_index, swap_rp)
    return indi


def softmutate(individual):
    """Soft constraint mutation
    Move a random lecture to a random room period
    if room period is already taken then swap room period with room period
    of old lecture"""

    num_lectures = len(cfg.LECTURELIST)
    num_roomtimes = len(cfg.ROOMPERIODLIST)

    rand_lect_index = random.randint(0, (num_lectures - 1))
    rand_rp_index = random.randint(0, (num_roomtimes - 1))

    if cfg.CHAINMOVE:
        newind = kempemove(rand_lect_index, rand_rp_index, individual)
    else:
        newind = simplemovelecture(rand_lect_index, rand_rp_index, individual)

    # for i in range(0, len(individual)):
    #     print(i, ": ", individual[i], newind[i])
    return newind,  # trailing comma important for single obj deap eval


def softtwopointcrossover(ind1, ind2):
    """Soft Constraint 2 point crossover
    move rt of event in ind1 with ind2"""

    c_slice = len(ind1) // 2  # get cross over point

    individual1 = copy.copy(ind1)
    individual2 = copy.copy(ind2)

    # print(c_slice, "cross over slice")

    # get new offspring from first part of crossover
    if cfg.CHAINMOVE:
        for i in range(c_slice):
            # individual1 = simplemovelecture(i, ind2[i], individual1)
            individual1 = kempemove(i, ind2[i], individual1)

        # get new offspring from second part of cross over
        for i in range(c_slice, len(ind1)):
            # individual2 = simplemovelecture(i, ind1[i], individual2)
            individual2 = kempemove(i, ind1[i], individual2)
    else:
        for i in range(c_slice):
            individual1 = simplemovelecture(i, ind2[i], individual1)
            # individual1 = kempemove(i, ind2[i], individual1)

        # get new offspring from second part of cross over
        for i in range(c_slice, len(ind1)):
            individual2 = simplemovelecture(i, ind1[i], individual2)
            # individual1 = kempemove(i, ind2[i], individual1)
    # for i in range(len(ind1)):
    #     print(i, ": ", ind1[i], ind2[i], individual1[i], individual2[i])
    # print("parent1")
    # print(ind1)
    # print("parent2")
    # print(ind2)
    # print("offspring1")
    # print(individual1)
    # print("offspring2")
    # print(individual2, "\n\n")
    return individual1, individual2


def softuniformcrossover(ind1, ind2):
    """hard crostraint uniform crossover"""
    numswaps = len(ind1) // 2

    individual1 = copy.copy(ind1)
    individual2 = copy.copy(ind2)

    swaps = random.sample(range(len(ind1)), numswaps)
    # print(swaps)

    if cfg.CHAINMOVE:
        for index in swaps:
            # individual1 = simplemovelecture(index, ind2[index], individual1)
            # individual2 = simplemovelecture(index, ind1[index], individual2)
            individual1 = kempemove(index, ind2[index], individual1)
            individual2 = kempemove(index, ind1[index], individual2)
    else:
        for index in swaps:
            individual1 = simplemovelecture(index, ind2[index], individual1)
            individual2 = simplemovelecture(index, ind1[index], individual2)
            # individual1 = kempemove(index, ind2[index], individual1)
            # individual2 = kempemove(index, ind1[index], individual2)
    #     print(index)

    # print(individual1)
    # print(individual2)
    return ind1, ind2


def softcrossover(ind1, ind2):
    """select type of cross over
    Uniform or two point cross over"""
    # if cfg.CURRENTGEN % 200 > 100:  # SWAP CROSSOVER TYPE EVERY 100 GENS
    #     return huniformcrossover(ind1, ind2)
    # else:
    #     return hcrossover(ind1, ind2)
    # if random.random() > 0.5:
    #     return softtwopointcrossover(ind1, ind2)
    # else:
    #     return softuniformcrossover(ind1, ind2)
    return softtwopointcrossover(ind1, ind2)


def softevaluation(individual):
    """hard constraints evaluation function for individual
    in population"""

    schedule = readectt.createschedule(
        individual, cfg.ROOMPERIODLIST, cfg.LECTURELIST, cfg.DATA[0])
    hcveval = readectt.gethardconstraintsviolations(schedule, cfg.DATA)
    hcveval *= 1000
    sceval = readectt.getsoftconstraintviolations(schedule, cfg.DATA)
    # trailing comma important for single obj deap eval
    fitness = hcveval * sceval + sceval,

    fileopen = open(cfg.FNAME, "a")
    for allele in individual:
        fileopen.write(str(allele))
        fileopen.write(",")
    # fileopen.write(str(next(iter(fitness))))
    fileopen.write(str(sceval + (hcveval/1000)))
    fileopen.write(",")
    if hcveval == 0:
        fileopen.write("F")
    else:
        fileopen.write("NF")
    fileopen.write("\n")
    fileopen.close()

    return fitness


# Utility functions needed for moves
def geteventssharingperiod(individual, rpindex):
    """ Get events in the period of specified Room Period object"""

    roomperiod = cfg.ROOMPERIODLIST[rpindex]
    per = roomperiod.period
    # print("Index", rpindex, " is ", roomperiod, " is in  period ", per)
    # print(sameperiodindex)
    sameperiods = [i for i, j in enumerate(cfg.ROOMPERIODLIST)
                   if cfg.ROOMPERIODLIST[i].period == per]

    eventsinperiod = [i for i, j in enumerate(
        individual) if individual[i] in sameperiods]

    return eventsinperiod


def getcurlectroom(lectureindex, individual):
    """Get room an event in scheduled in, curricula event
    belongs to and lecturer taking the event"""

    course = cfg.LECTURELIST[lectureindex]
    # print("\tEvent name is ", course)

    courselecturerlist = cfg.DATA[1][1]
    lecturer = next((item for item in courselecturerlist if item[
        "coursecode"] == course))
    # print("\tLecturer teaching is:", lecturer['lecturer'])

    coursecurricula = cfg.DATA[3][1]
    # print("\tstudent curricula taking is: ", coursecurricula[course])

    rp_index = individual[lectureindex]
    room_period = cfg.ROOMPERIODLIST[rp_index]
    # print("\tLecture is taught in: ", room_period)

    return (course,
            lecturer['lecturer'],
            room_period.roomid,
            coursecurricula[course])


def movesinglekempe(individual, lect_index, avail_rooms, period):
    """Single move to a period, if period is taken
    then move to random available period"""

    # print(avail_rooms, pref_room, "kempe")
    # print(pref_room)
    if len(avail_rooms) == 0:
        return individual
    # print(avail_rooms)
    # print(cfg.LECTURELIST[lect_index])
    # DATA[1][0]
    course = next(x for x in cfg.DATA[1][0]
                  if x.coursecode == cfg.LECTURELIST[lect_index])
    # print(cfg.DATA[2][1])
    # rooms = []
    # for venue in avail_rooms:
    #     venue_obj = next(x for x in cfg.DATA[1][0]
    #                      if x.roomid == venue)
    # print(course)
    room = list(avail_rooms)[0]
    best_diff = course.num_stud
    for _, val in enumerate(avail_rooms):
        # get room
        roomsize = int(cfg.DATA[2][1][val])
        diff = course.num_stud - roomsize
        # temp = best_diff
        # print("\t", val)
        # room that can fit more students taking csc
        if diff < best_diff or diff <= 0:
            if best_diff <= 0 and diff <= 0:
                # try to get perfect fit -- room not too big for event
                if abs(diff) < abs(best_diff):
                    room = val
                    best_diff = diff
                    # print("\t\tChosing from absolute", temp, diff)
                # else:
                    # print("\t\tchosing from absolute", temp, diff)
            else:
                room = val
                best_diff = diff
                # print("\t\tChosing from simple less", temp, diff)
        # else:
            # print("\t\tNot entering", temp, diff)
    # print("\t\tRoom chosen", room)
    new_index = [i for i, j in enumerate(cfg.ROOMPERIODLIST)
                 if cfg.ROOMPERIODLIST[i].period == period and
                 cfg.ROOMPERIODLIST[i].roomid == room]

    ind = list(individual)
    if isinstance(new_index[0], int):
        ind[lect_index] = new_index
        avail_rooms.remove(room)
        individual[lect_index] = new_index[0]

    # print("taking ", room, cfg.ROOMPERIODLIST[new_index[0]].roomid,
    #       cfg.ROOMPERIODLIST[new_index[0]].period)

    return ind


def kempemove(lecture_index, new_rp, ind):
    """Kempe move event to new room period
    Kempe means clashing events (curricula, room, lecturer) in new period are
    moved to old period. Rooms are assigned to random available"""

    individual = copy.copy(ind)
    old_rp_index = individual[lecture_index]

    to_move_event = getcurlectroom(lecture_index, individual)

    try:
        new_rp_index = individual.index(new_rp)
    except ValueError:
        num_roomtimes = len(cfg.ROOMPERIODLIST)
        new_rp_index = random.randint(0, (num_roomtimes - 1))

    # print("\n\n new period event to be moved to")
    events_in_new_per = geteventssharingperiod(individual, new_rp_index)
    events_in_old_per = geteventssharingperiod(individual, old_rp_index)

    csc_in_newp = []
    avail_rooms_np = set(i.roomid for i in cfg.DATA[2][0])
    avail_rooms_ol = set(i.roomid for i in cfg.DATA[2][0])

    for event in events_in_new_per:
        clr = getcurlectroom(event, individual)
        csc_in_newp.append(clr[0])
        # remove taken rooms in current
        try:
            avail_rooms_np.remove(clr[2])
        except KeyError:
            pass

    if to_move_event[0] in csc_in_newp:
        # print("Event already in period - no move", to_move_event[0])
        return individual

    old_index_move = set()
    for item in events_in_new_per:
        new_period_event = getcurlectroom(item, individual)
        cur_lect_new_per = set(new_period_event[3])
        # add event to move list if curr clash
        if len(set(to_move_event[3]).intersection(cur_lect_new_per)) > 0:
            old_index_move.add(item)
        # add event to move list if lect clash
        if new_period_event[1] == to_move_event[1]:
            old_index_move.add(item)
        # add event to move list if lect clash
        # if new_period_event[2] == to_move_event[2]:
        #     old_index_move.add(item)

    # print("\n")

    # get clash curricula and lectures
    lect_in_old_index = []
    cur_in_old_index = []

    for event in old_index_move:
        clr = getcurlectroom(event, individual)
        lect_in_old_index.append(clr[1])
        avail_rooms_np.add(clr[2])
        for cur in clr[3]:
            cur_in_old_index.append(cur)

    new_index_move = {lecture_index}
    # add other events that will cause a curr
    # or lect clash by old moves
    for item in events_in_old_per:
        period_event = getcurlectroom(item, individual)
        cur_lect_per = set(period_event[3])
        # remove taken rooms
        try:
            avail_rooms_ol.remove(period_event[2])
        except KeyError:
            pass
        moving_event = False
        # add event to move list if curr clash
        if len(set(cur_in_old_index).intersection(cur_lect_per)) > 0:
            new_index_move.add(item)
            moving_event = True
        # add event to move list if lect clash
        if period_event[1] in lect_in_old_index:
            new_index_move.add(item)
            moving_event = True
        if moving_event:
            avail_rooms_ol.add(period_event[2])

    old_period = cfg.ROOMPERIODLIST[old_rp_index].period
    new_period = cfg.ROOMPERIODLIST[new_rp_index].period

    # kempe move old lectures
    for evt in new_index_move:
        period_event = getcurlectroom(evt, individual)
        # print("old_period", cfg.ROOMPERIODLIST[individual[evt]].period)
        movesinglekempe(individual, evt,
                        avail_rooms_np, new_period)

    for evt in old_index_move:
        period_event = getcurlectroom(evt, individual)
        # print("old_period", cfg.ROOMPERIODLIST[individual[evt]].period)
        movesinglekempe(individual, evt,
                        avail_rooms_ol, old_period)
    return individual

# geteventssharingperiod()
