"""Main entry point for creating timetable"""

import argparse
from os.path import abspath, join, isfile

import random
import pickle
# from time import time
import numpy
# import copy
# import json

import readectt
import genetic_operators
import config as cfg

from deap import base
from deap import algorithms
from deap import creator
from deap import tools


NGEN = 10000
# NGEN = 1
MU = 400
LAMBDA = 150
CXPB = 0.7
MUTPB = 0.2
RANDSEED = 64
FREQ = 50  # save checkpoint ever 50 generations
MAXGENNOINPROVE = 20


def parse_args():
    """Parse command line arguments passed to script invocation."""
    parser = argparse.ArgumentParser(
        description='Create violation free timetable solution')

    parser.add_argument('filename', help='Timetable problem input file')

    return parser.parse_args()


def initindividual(icls, content):
    """Initialize an individual"""
    return icls(content)


def initpopulation(pcls, ind_init, ind_list):
    """Initilize a population from list"""
    return pcls(ind_init(c) for c in ind_list)


def softconstenvolve(pop_hard, checkpoint=None):
    """Evolve population with soft constraint operators"""
    # for ind in pop:
    #     print(ind, "\n", genetic_operators.hevaluation(ind))
    pop = convertpop(pop_hard)

    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("individual", initindividual, creator.Individual)
    toolbox.register("population", initpopulation, list,
                     toolbox.individual, pop)

    toolbox.register("evaluate", genetic_operators.softevaluation)

    toolbox.register("mutate", genetic_operators.softmutate)
    toolbox.register("mate", genetic_operators.softcrossover)
    # toolbox.register("mate", genetic_operators.huniformcrossover)
    # toolbox.register("select", tools.selNSGA2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return evolve(toolbox, checkpoint, False)


def evolve(toolbox, checkpoint, evolve_type=True):
    """Evolve a population and return final population
    eaMuPlusLambda implementation with saving checkpoints
    stop criteria at minimum or consecutive non improve

    If evolve type is True then hardconstraint evolve
    else soft constraint evolve"""
    # pop = toolbox.population(n=MU)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    if isfile(checkpoint):
        # A filename that has been given exists,then load data from the file
        with open(checkpoint, "rb") as cp_file:
            cpoint = pickle.load(cp_file)
        pop = cpoint["population"]
        gen = cpoint["generation"]
        halloffame = cpoint["halloffame"]
        consecutive = cpoint["consecutive"]
        logbook = cpoint["logbook"]
        random.setstate(cpoint["rndstate"])
        if "chainmove" in cpoint:
            cfg.CHAINMOVE = cpoint["chainmove"]
        # print(cpoint["consecutive"], cpoint["generation"])
    else:
        # Start a new evolution
        if evolve_type:
            pop = toolbox.population(n=MU)
        else:
            pop = toolbox.population()
        gen = 0
        consecutive = 0
        halloffame = tools.HallOfFame(maxsize=1)
        logbook = tools.Logbook()
        cfg.CHAINMOVE = False
        logbook.header = ['gen', 'nevals'] + stats.fields

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # log generation 0
    halloffame.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    fits = [ind.fitness.values[0] for ind in pop]
    currentmin = min(fits)
    if gen == 0:
        gen += 1

    # Begin the generational process
    # for gen in range(1, NGEN + 1)
    while gen < NGEN and currentmin > 0 and consecutive < MAXGENNOINPROVE:
        # Vary the population
        cfg.CURRENTGEN = gen
        offspring = algorithms.varOr(pop, toolbox, LAMBDA, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness - No fit val
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop[:] = toolbox.select(pop + offspring, MU)

        # Update records and log book
        halloffame.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        # check for no improvement
        if currentmin <= min(fits):
            consecutive += 1
        else:
            consecutive = 0

        # toggle between chain move or simple move
        if consecutive == 120:
            cfg.CHAINMOVE = not cfg.CHAINMOVE

        currentmin = min(fits)
        # save checkpoint
        if gen % FREQ == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cpoint = dict(population=pop, generation=gen,
                          halloffame=halloffame,
                          logbook=logbook,
                          chainmove=cfg.CHAINMOVE,
                          rndstate=random.getstate(),
                          consecutive=consecutive)

            with open(checkpoint, "wb") as cp_file:
                pickle.dump(cpoint, cp_file)

        print(logbook.stream)  # print records for generation
        # print(currentmin, consecutive)
        gen += 1

    print("-- End of (successful) evolution --")

    # Fill the dictionary using the dict(key=value[, ...]) constructor
    # save last state before exit
    cpoint = dict(population=pop, generation=gen,
                  halloffame=halloffame,
                  logbook=logbook, rndstate=random.getstate(),
                  chainmove=cfg.CHAINMOVE,
                  consecutive=consecutive)

    with open(checkpoint, "wb") as cp_file:
        pickle.dump(cpoint, cp_file)

    # return halloffame and final population
    return halloffame, pop


def hardconstevolve(checkpoint=None):
    """Evolve solve hard constraints and return solution"""
    # minimizing fitness genetic algorithm
    random.seed(64)
    # random.seed(time())
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # individual with random integer allele - unique elements
    ind_size = len(cfg.LECTURELIST)
    # poss_allele = len(cfg.ROOMPERIODLIST)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(ind_size), ind_size)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.indices)
    # register population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", genetic_operators.hardevaluation)
    toolbox.register("mutate", genetic_operators.hardmutate)
    toolbox.register("mate", genetic_operators.hardtwopointcrossover)
    # toolbox.register("mate", genetic_operators.huniformcrossover)
    # toolbox.register("select", tools.selNSGA2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return evolve(toolbox, checkpoint)


def convertpop(population):
    """Convert individuals in population for soft constraints evolution

    Hard constraint individual is a list of unique integers to maximum
    number of lectures. Index i is the order individual[i] will be assigned
    a room and period. individual[i] is the index of an event in the
    event list (lecturelist)

    Soft constraint individual is a list of unique intigers to maximun number
    of possible roomperiod - num rooms * number of periods. Index i is the
    index of the event in the event list (lecturelist) while individual[i]
    is the index of the room and period in the roomperiod list.
    """
    print("Converting hard constraint pop to soft constraint pop ....")
    pop = []
    for _, individual in enumerate(population):
        # print(i, individual, len(individual), "\n")
        # create human understandable timetable
        # from hard constraint individual
        temp_schedule = readectt.constructschedule(
            individual, cfg.ROOMPERIODLIST, cfg.LECTURELIST, cfg.DATA)
        # print(readectt.gethardconstraintsviolations(
        #     temp_schedule, cfg.DATA), "hard")

        # convert a human understandable timetable
        # to a soft constraint individual
        soft_indi = readectt.scheduletosolution(temp_schedule,
                                                cfg.ROOMPERIODLIST,
                                                cfg.LECTURELIST)
        # convert softconstraint individual to
        # human understandable timetable
        # conv_sch = readectt.createschedule(
        #     soft_indi, cfg.ROOMPERIODLIST, cfg.LECTURELIST, cfg.DATA[0])
        # print(readectt.gethardconstraintsviolations(
        #     conv_sch, cfg.DATA), "soft")
        # print(i, soft_indi, len(individual), "converted individual")
        pop.append(soft_indi)
    return pop


def main():
    """Module's main entry point (zopectl.command)."""

    # Reading input file - Timetable problem and saving in global variable
    args = parse_args()
    filename = args.filename
    inputfilename = filename + '.ectt'
    solutionname = filename + '.sol'
    fullpath = join(abspath('../InputData/ITC-2007_ectt'), inputfilename)

    cfg.FNAME = join(abspath('../ML-Training-Data'), (filename + '.data'))

    if not isfile(fullpath):
        raise Exception('Directory does not exist ({0}).'.format(fullpath))

    cfg.DATA = list(readectt.readecttfile(fullpath))
    cfg.ROOMPERIODLIST = readectt.createroomperiodlist(
        cfg.DATA[2][0], cfg.DATA[0].getnumperiods())
    cfg.LECTURELIST = readectt.createlecturelist(
        cfg.DATA[1][0])  # Course list [1][1] is lecturer

    # DATA[0]  "HEADER"
    # DATA[1][0]  "COURSES"
    # DATA[1][1]  "COURSELLECTURERLIST"
    # DATA[1][2]  "COURSECOUNTLIST"
    # DATA[1][3]  "COURSE SIZE"
    # DATA[1][3]  "COURSE Min working day"
    # DATA[2][0]  "ROOMS LIST"
    # DATA[2][1]  "ROOMS SIZE DICT"
    # DATA[2][2]  "ROOMS SITE DICT"
    # DATA[3][0]  "CURRICULA"
    # DATA[3][1]  "CURRICULA TAKING COURSE DICT OF {COURSE:[CUR1, CUR2]}"
    # DATA[4][0]  "UNAVAILABLE CONSTRAINTS"
    # DATA[4][1]  "COURSE UNVAIL PERIOD DICT"
    # DATA[5]  "ROOM CONSTRAINTS"

    # randsol = readectt.createrandomsolution(
    #     len(cfg.LECTURELIST), len(cfg.ROOMPERIODLIST))
    # # for i, course in enumerate(cfg.DATA[1][0]):
    # #     print(course, i)

    # genetic_operators.kempemove(50, 5, randsol)
    # return

    # print(cfg.DATA[4][1])
    hardcheckpointname = filename + '.cph'  # file to store GA checkpoints
    cph_fullpath = join(abspath('../Checkpoints'), hardcheckpointname)

    softcheckpointname = filename + '.cps'  # file to store GA checkpoints
    cps_fullpath = join(abspath('../Checkpoints'), softcheckpointname)

    # randsol = readectt.createconstructsequece(cfg.LECTURELIST)

    # schedule = readectt.constructschedule(
    #     randsol, cfg.ROOMPERIODLIST, cfg.LECTURELIST, cfg.DATA)

    # softconstenvolve(finalpop)

    # hard constraint evolution
    _, pop_hard = hardconstevolve(cph_fullpath)

    # soft constraint evolution
    hof, _ = softconstenvolve(pop_hard, cps_fullpath)

    schedule = readectt.createschedule(
        hof[0], cfg.ROOMPERIODLIST, cfg.LECTURELIST, cfg.DATA[0])
    readectt.writescheduletofile(schedule, solutionname)
    print("Hard Constraint Violations")
    hcv = readectt.gethardconstraintsviolations(schedule, cfg.DATA)
    print("\tTotal: ", hcv, "\n")
    print("Soft Constraint Violations")
    scv = readectt.getsoftconstraintviolations(schedule, cfg.DATA)
    print("Total: ", scv, "\n")


if __name__ == '__main__':
    main()
    # l_pop, l_log, hofame = main()
    # for item in hofame:
    #     print(item, "\n")
    #     schedule = readectt.createschedule(
    #         item, cfg.ROOMPERIODLIST, cfg.LECTURELIST, cfg.DATA[0])
    #     # readectt.writescheduletofile(schedule, 'comp01.sol')
    #     print("Hard Constraint Violations")
    #     hcv = readectt.gethardconstraintsviolations(schedule, cfg.DATA)
    #     print("\tTotal: ", hcv, "\n\n")
