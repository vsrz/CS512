import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions
from sklearn import svm  # provides Support Vector Regression
import csv
import math
import sys

# Local files created by me
import FromDataFileSVM_GA
import FromFinessFileSVM_GA

#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def writeTheHeader():
    with open('GA-SVR.csv', 'ab') as csvfile:
        modelwriter = csv.writer(csvfile)
        modelwriter.writerow(['Descriptor Ids', 'Num of desc',
                              'Fitness', 'RMSE', 'TrainR2', 'RMSEValidate',
                              'ValidateR2', 'TestR2', 'Model', 'Localtime'])


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def getAValidrow(numOfFea, eps=0.015):
    sum = 0;
    while (sum < 3):
        V = zeros(numOfFea)
        for j in range(numOfFea):
            r = random.uniform(0, 1)
            if (r < eps):
                V[j] = 1
            else:
                V[j] = 0
        sum = V.sum()
    return V


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def createPopulationArray(numOfPop, numOfFea):
    population = random.random((numOfPop, numOfFea))
    for i in range(numOfPop):
        V = getAValidrow(numOfFea)
        for j in range(numOfFea):
            population[i][j] = V[j]
    return population


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def findSecondElite(elite1Index, fitness, population):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]

    elite2 = zeros(numOfFea)
    elite2Index = 0
    if (elite1Index == elite2Index):
        elite2Index = 1

    for i in range(elite2Index, numOfPop):
        if (i <> elite1Index) and (fitness[i] <= fitness[elite2Index]):
            elite2Index = i

    for j in range(numOfFea):
        elite2[j] = population[elite2Index][j]

    return elite2, elite2Index


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def findFirstElite(fitness, population):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    elite1 = zeros(numOfFea)
    elite1Index = 0

    for i in range(1, numOfPop):
        if (fitness[i] < fitness[elite1Index]):
            elite1Index = i

    for j in range(numOfFea):
        elite1[j] = population[elite1Index][j]

    return elite1, elite1Index


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def findElites(fitness, population):
    elite1, elite1Index = findFirstElite(fitness, population)
    elite2, elite2Index = findSecondElite(elite1Index, fitness, population)
    return elite1, elite2, elite1Index, elite2Index


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def AddFitnessForFindingParents(fitness):
    numOfPop = fitness.shape[0]
    sumFitnesses = zeros(numOfPop)
    sumFitnesses[0] = fitness[0]
    for i in range(1, numOfPop):
        sumFitnesses[i] = fitness[i] + sumFitnesses[i - 1]

    return sumFitnesses


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def OnePointCrossOver(mom, dad):
    numOfFea = mom.shape[0]
    p = int(random.uniform(1, numOfFea - 1))
    child = zeros(numOfFea)

    for j in range(p):
        child[j] = mom[j]
    for j in range(p - 1, numOfFea):
        child[j] = dad[j]

    return child


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def find1st2ndAnd3rdPoints(numOfFea):
    point1 = int(random.uniform(0, numOfFea))
    point2 = int(random.uniform(0, numOfFea))
    while (point1 == point2):
        point2 = int(random.uniform(0, numOfFea))
    point3 = int(random.uniform(0, numOfFea))
    while ((point1 == point3) or (point2 == point3)):
        point3 = int(random.uniform(0, numOfFea))
    a = [point1, point2, point3]
    a.sort()
    p1 = a[0]
    p2 = a[1]
    p3 = a[2]
    return p1, p2, p3


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def find1stAnd2ndPoints(numOfFea):
    point1 = int(random.uniform(0, numOfFea))
    point2 = point1
    while (point1 == point2):
        point2 = int(random.uniform(0, numOfFea))
    if (point2 < point1):
        p1 = point2
        p2 = point1
    else:
        p1 = point1
        p2 = point2
    return p1, p2


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def TwoPointsCrossOver(mom, dad):
    numOfFea = mom.shape[0]
    p1, p2 = find1stAnd2ndPoints(numOfFea)
    child = zeros(numOfFea)
    for j in range(numOfFea):
        child[j] = mom[j]

    for i in range(p1 - 1, p2):
        child[i] = dad[i]

    return child


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def ThreePointsCrossOver(mom, dad):
    numOfFea = mom.shape[0]
    p1, p2, p3 = find1st2ndAnd3rdPoints(numOfFea)
    child = zeros(numOfFea)
    for j in range(numOfFea):
        child[j] = mom[j]

    for i in range(p1 - 1, p2):
        child[i] = dad[i]
    for i in range(p3 - 1, numOfFea):
        child[i] = dad[i]

    return child


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def mutate(child):
    numOfFea = child.shape[0]
    for i in range(numOfFea):
        p = random.uniform(0, 100)
        child[i] = 1 - child[i]
    return child


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013
# uses the Roulette Wheel methed to choose an index. So the 
# ones with higher probablilities are more likely to be chosen

def chooseAnIndexFromPopulation(sumOfFitnesses, population):
    numOfPop = sumOfFitnesses.shape[0]
    p = random.uniform(0, sumOfFitnesses[numOfPop - 1])
    i = 0
    while (p > sumOfFitnesses[i]) and (i < (numOfPop - 1)):
        i = i + 1
    if (i < numOfPop):
        return i
    else:
        return numOfPop - 1


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def findTheParents(sumOfFitnesses, population):
    numOfFea = population.shape[1]
    momIndex = chooseAnIndexFromPopulation(sumOfFitnesses, population)
    dadIndex = chooseAnIndexFromPopulation(sumOfFitnesses, population)
    while (dadIndex == momIndex):
        dadIndex = chooseAnIndexFromPopulation(sumOfFitnesses, population)
    dad = zeros(numOfFea)
    mom = zeros(numOfFea)
    for j in range(numOfFea):
        dad[j] = population[dadIndex][j]
        mom[j] = population[momIndex][j]
    return mom, dad


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def findTheChild(mom, dad):
    child = OnePointCrossOver(mom, dad)
    #child = TwoPointsCrossOver(mom, dad)
    #child = ThreePointsCrossOver(mom, dad)
    child = mutate(child)
    return child


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def equal(child, popI):
    numOfFea = child.shape[0]
    for j in range(numOfFea):
        if (child[j] <> popI[j]):
            return 0
    return 1


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def IsChildUnique(upToRowI, child, population):
    numOfFea = child.shape[0]
    for i in range(upToRowI):
        for j in range(numOfFea):
            if (equal(child, population[i])):
                return 0
    return 1


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def findNewPopulation(elite1, elite2, sumOfFitnesses, population):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    for j in range(numOfFea):
        population[0][j] = elite1[j]
        population[1][j] = elite2[j]

    for i in range(2, numOfPop):
        uniqueRow = 0
        sum = 0;
        while (sum < 3) or (not uniqueRow):
            mom, dad = findTheParents(sumOfFitnesses, population)
            child = findTheChild(mom, dad)
            uniqueRow = IsChildUnique(i, child, population)
            sum = child.sum()
        for k in range(numOfFea):
            population[i][k] = child[k]

    return population


#-----------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def createInitialPopulation(numOfPop, numOfFea):
    population = random.random((numOfPop, numOfFea))
    for i in range(numOfPop):
        V = getAValidrow(numOfFea)
        for j in range(numOfFea):
            population[i][j] = V[j]
    return population


#------------------------------------------------------------------------------
def checkterTerminationStatus(Times, oldFitness, minimumFitness):
    if (Times == 30):
        print "***** No need to continue. The fitness not changed in the last 30 generation"
        exit(0)
    elif (oldFitness == minimumFitness):
        Times = Times + 1
    elif (minimumFitness < oldFitness):
        oldFitness = minimumFitness
        Times = 0
        print "\n***** time is = ", time.strftime("%H:%M:%S", time.localtime())
        print "******************** Times is set back to 0 ********************\n"
    return oldFitness, Times


#------------------------------------------------------

#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def IterateNtimes(model, fileW, fitness, sumOfFitnesses, population,
                  elite1, elite2, elite1Index, elite2Index,
                  TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
    unfit = 1000
    numOfGenerations = 2000  # should be 2000
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    Times = 0
    oldFitness = fitness.min()
    for i in range(1, numOfGenerations):
        oldFitness, Times = checkterTerminationStatus(Times, oldFitness, fitness.min())
        print "This is generation ", i, " --- Minimum of fitness is: -> ", fitness.min()
        if (fitness.min() < 0.005):
            print "***********************************"
            print "Good: Fitness is low enough to quit"
            print "***********************************"
            exit(0)
        fittingStatus = unfit
        while (fittingStatus == unfit):
            population = findNewPopulation(elite1, elite2, sumOfFitnesses, population)
            fittingStatus, fitness = FromFinessFileSVM_GA.validate_model(model, fileW,
                                                                         population, TrainX, TrainY, ValidateX,
                                                                         ValidateY, TestX, TestY)

        elite1, elite2, elite1Index, elite2Index = findElites(fitness, population)
        #adding all the fitnesses and storing them in one dimensional array for
        #choosing the children for the next round
        sumFitnesses = AddFitnessForFindingParents(fitness)

    return


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 18, 2013
def createAnOutputFile():
    file_name = None
    algorithm = None
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if ( (file_name == None) and (algorithm != None)):
        file_name = "{}_{}_gen{}_{}.csv".format(alg.__class__.__name__,
                                                alg.model.__class__.__name__, alg.gen_max, timestamp)
    elif file_name == None:
        file_name = "{}.csv".format(timestamp)
    fileOut = file(file_name, 'wb')
    fileW = csv.writer(fileOut)

    fileW.writerow(['Descriptor ID', 'No. Descriptors', 'Fitness', 'Model', 'R2', 'Q2', \
                    'R2Pred_Validation', 'R2Pred_Test', 'SEE_Train', 'SDEP_Validation', 'SDEP_Test', \
                    'y_Train', 'yHat_Train', 'yHat_CV', 'y_validation', 'yHat_validation', 'y_Test', 'yHat_Test'])

    return fileW


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

#main program starts in here
def main():
    fileW = createAnOutputFile()
    model = svm.SVR()

    numOfPop = 50  # should be 200 population, lower is faster but less accurate
    numOfFea = 385  # should be 385 descriptors
    unfit = 1000

    # Final model requirements
    R2req_train = .6
    R2req_validate = .5
    R2req_test = .5

    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileSVM_GA.getAllOfTheData()
    TrainX, ValidateX, TestX = FromDataFileSVM_GA.rescaleTheData(TrainX, ValidateX, TestX)

    #numOfFea = TrainX.shape[1]  # should be 396 descriptors

    unfit = 1000  # when to stop when the model isn't doing well

    fittingStatus = unfit
    while (fittingStatus == unfit):
        population = createInitialPopulation(numOfPop, numOfFea)
        fittingStatus, fitness = FromFinessFileSVM_GA.validate_model(model, fileW, population,
                                                                     TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    elite1, elite2, elite1Index, elite2Index = findElites(fitness, population)
    sumOfFitnesses = AddFitnessForFindingParents(fitness)

    print "Starting the Loop - time is = ", time.strftime("%H:%M:%S", time.localtime())
    IterateNtimes(model, fileW, fitness, sumOfFitnesses, population,
                  elite1, elite2, elite1Index, elite2Index,
                  TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    return

#main program ends in here
#------------------------------------------------------------------------------
main()
#------------------------------------------------------------------------------



