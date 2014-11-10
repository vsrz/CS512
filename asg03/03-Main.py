import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions
from sklearn import svm  # provides Support Vector Regression
from sklearn import linear_model
import csv
import math
import sys
import hashlib

def r2(y, yHat):
    """Coefficient of determination"""
    numer = ((y - yHat) ** 2).sum()  # Residual Sum of Squares
    denom = ((y - y.mean()) ** 2).sum()
    r2 = 1 - numer / denom
    return r2

def r2Pred(yTrain, yTest, yHatTest):
    numer = ((yHatTest - yTest) ** 2).sum()
    denom = ((yTest - yTrain.mean()) ** 2).sum()
    r2Pred = 1 - numer / denom
    return r2Pred

def see(p, y, yHat):
    """
    Standard error of estimate
    (Root mean square error)
    """
    n = y.shape[0]
    numer = ((y - yHat) ** 2).sum()
    denom = n - p - 1
    if (denom == 0):
        s = 0
    elif ( (numer / denom) < 0 ):
        s = 0.001
    else:
        s = (numer / denom) ** 0.5
    return s


#------------------------------------------------------------------------------

def sdep(y, yHat):
    """
    Standard deviation of error of prediction
    (Root mean square error of prediction)
    """
    n = y.shape[0]

    numer = ((y - yHat) ** 2).sum()

    sdep = (numer / n) ** 0.5

    return sdep

def cv_predict(set_x, set_y, model):
    """Predict using cross validation."""
    yhat = empty_like(set_y)
    for idx in range(0, yhat.shape[0]):
        train_x = delete(set_x, idx, axis=0)
        train_y = delete(set_y, idx, axis=0)
        model = model.fit(train_x, train_y)
        yhat[idx] = model.predict(set_x[idx])
    return yhat

def calc_fitness(xi, Y, Yhat, c=2):
    """
    Calculate fitness of a prediction.

    Parameters
    ----------
    xi : array_like -- Mask of features to measure fitness of. Must be of dtype bool.
    model : object  --  Object to make predictions, usually a regression model object.
    c : float       -- Adjustment parameter.

    Returns
    -------
    out: float -- Fitness for the given data.

    """

    p = sum(xi)  # Number of selected parameters
    n = len(Y)  # Sample size
    numer = ((Y - Yhat) ** 2).sum() / n  # Mean square error
    pcn = p * (c / n)
    if pcn >= 1:
        return 1000
    denom = (1 - pcn) ** 2
    theFitness = numer / denom
    return theFitness

def InitializeTracks():
    trackDesc = {}
    trackIdx = {}
    trackFitness = {}
    trackModel = {}
    trackR2 = {}
    trackQ2 = {}
    trackR2PredValidation = {}
    trackR2PredTest = {}
    trackSEETrain = {}
    trackSDEPValidation = {}
    trackSDEPTest = {}
    return trackDesc, trackIdx, trackFitness, trackModel, trackR2, trackQ2, \
           trackR2PredValidation, trackR2PredTest, trackSEETrain, \
           trackSDEPValidation, trackSDEPTest

def initializeYDimension():
    yTrain = {}
    yHatTrain = {}
    yHatCV = {}
    yValidation = {}
    yHatValidation = {}
    yTest = {}
    yHatTest = {}
    return yTrain, yHatTrain, yHatCV, yValidation, yHatValidation, yTest, yHatTest

def OnlySelectTheOnesColumns(popI):
    numOfFea = popI.shape[0]
    xi = zeros(numOfFea)
    for j in range(numOfFea):
        xi[j] = popI[j]

    xi = xi.nonzero()[0]
    xi = xi.tolist()
    return xi

def validate_model(model, population, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
    numOfPop = population.shape[0]
    fitness = zeros(numOfPop)
    c = 2
    false = 0
    true = 1
    predictive = false

    trackDesc, trackIdx, trackFitness, trackModel, trackR2, trackQ2, \
    trackR2PredValidation, trackR2PredTest, trackSEETrain, \
    trackSDEPValidation, trackSDEPTest = InitializeTracks()

    yTrain, yHatTrain, yHatCV, yValidation, \
    yHatValidation, yTest, yHatTest = initializeYDimension()

    unfit = 1000
    itFits = 1
    for i in range(numOfPop):
        xi = OnlySelectTheOnesColumns(population[i])

        idx = hashlib.sha1(array(xi)).digest()  # Hash
        if idx in trackFitness.keys():
            # don't recalculate everything if the model has already been validated
            fitness[i] = trackFitness[idx]
            continue

        X_train_masked = TrainX.T[xi].T
        X_validation_masked = ValidateX.T[xi].T
        X_test_masked = TestX.T[xi].T

        try:
            model_desc = model.fit(X_train_masked, TrainY)
        except:
            return unfit, fitness

        # Computed predicted values
        Yhat_cv = cv_predict(X_train_masked, TrainY, model)  # Cross Validation
        Yhat_validation = model.predict(X_validation_masked)
        Yhat_test = model.predict(X_test_masked)

        # Compute R2 statistics (Prediction for Validation and Test set)
        q2_loo = r2(TrainY, Yhat_cv)
        r2pred_validation = r2Pred(TrainY, ValidateY, Yhat_validation)
        r2pred_test = r2Pred(TrainY, TestY, Yhat_test)

        Y_fitness = append(TrainY, ValidateY)
        Yhat_fitness = append(Yhat_cv, Yhat_validation)

        fitness[i] = calc_fitness(xi, Y_fitness, Yhat_fitness, c)

        #print "predictive is: ", predictive
        if predictive and ((q2_loo < 0.5) or (r2pred_validation < 0.5) or (r2pred_test < 0.5)):
            # if it's not worth recording, just return the fitness
            print "ending the program because of predictive is: ", predictive
            continue

        # Compute predicted Y_hat for training set.
        Yhat_train = model.predict(X_train_masked)
        r2_train = r2(TrainY, Yhat_train)

        # Standard error of estimate
        s = see(X_train_masked.shape[1], TrainY, Yhat_train)
        sdep_validation = sdep(ValidateY, Yhat_validation)
        sdep_test = sdep(TrainY, Yhat_train)

        idxLength = len(xi)

        # store stats
        trackDesc[idx] = str(xi)
        trackIdx[idx] = idxLength
        trackFitness[idx] = fitness[i]

        trackModel[idx] = model_desc

        trackR2[idx] = r2_train
        trackQ2[idx] = q2_loo
        trackR2PredValidation[idx] = r2pred_validation
        trackR2PredTest[idx] = r2pred_test
        trackSEETrain[idx] = s
        trackSDEPValidation[idx] = sdep_validation
        trackSDEPTest[idx] = sdep_test

        yTrain[idx] = TrainY.tolist()
        yHatTrain[idx] = Yhat_train.tolist()
        yHatCV[idx] = Yhat_cv.tolist()
        yValidation[idx] = ValidateY.tolist()
        yHatValidation[idx] = Yhat_validation.tolist()
        yTest[idx] = TestY.tolist()
        yHatTest[idx] = Yhat_test.tolist()
    return itFits, fitness

def placeDataIntoArray(fileName):
    with open(fileName, mode='rbU') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar=' ')
        dataArray = array([row for row in datareader], dtype=float64, order='C')

    if (min(dataArray.shape) == 1):  # flatten arrays of one row or column
        return dataArray.flatten(order='C')
    else:
        return dataArray


def getAllOfTheData():
    TrainX = placeDataIntoArray('Train-Data.csv')
    TrainY = placeDataIntoArray('Train-pIC50.csv')
    ValidateX = placeDataIntoArray('Validation-Data.csv')
    ValidateY = placeDataIntoArray('Validation-pIC50.csv')
    TestX = placeDataIntoArray('Test-Data.csv')
    TestY = placeDataIntoArray('Test-pIC50.csv')
    return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY


def rescaleTheData(TrainX, ValidateX, TestX):
    # 1 degree of freedom means (ddof) N-1 unbiased estimation
    TrainXVar = TrainX.var(axis=0, ddof=1)
    TrainXMean = TrainX.mean(axis=0)

    for i in range(0, TrainX.shape[0]):
        TrainX[i, :] = (TrainX[i, :] - TrainXMean) / sqrt(TrainXVar)
    for i in range(0, ValidateX.shape[0]):
        ValidateX[i, :] = (ValidateX[i, :] - TrainXMean) / sqrt(TrainXVar)
    for i in range(0, TestX.shape[0]):
        TestX[i, :] = (TestX[i, :] - TrainXMean) / sqrt(TrainXVar)

    return TrainX, ValidateX, TestX

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

def OnePointCrossOver(mom, dad):
    numOfFea = mom.shape[0]
    p = int(random.uniform(0, numOfFea - 1))
    child = zeros(numOfFea)

    for j in range(p):
        child[j] = mom[j]
    for j in range(p - 1, numOfFea):
        child[j] = dad[j]

    return child

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

def mutate(pop):
    numOfFea = pop.shape[0]
    for i in range(numOfFea):
        p = random.uniform(0, 100)
        # 5% chance of mutation on feature
        if (p < 0.05):
            pop[i] = abs(pop[i] - 1)
    return pop

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

def equal(child, popI):
    numOfFea = child.shape[0]
    for j in range(numOfFea):
        if (child[j] <> popI[j]):
            return 0
    return 1

def IsChildUnique(upToRowI, child, population):
    numOfFea = child.shape[0]
    for i in range(upToRowI):
        for j in range(numOfFea):
            if (equal(child, population[i])):
                return 0
    return 1

def findNewPopulation(elite1, sumOfFitnesses, population):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]

    for j in range(numOfFea):
        population[0][j] = elite1[j]

    for i in range(1, numOfPop):
        rnd = getThreeRandomRows(numOfPop,i)
        mut = []
        pop = []
        for e in range(0, 3):
            mut.append(population[rnd[e]])
            pop.append(mutate(mut[e]))
        population[i] = pop[random.randint(0,2)]
    return population

def createInitialPopulation(numOfPop, numOfFea):
    population = random.random((numOfPop, numOfFea))
    for i in range(numOfPop):
        V = getAValidrow(numOfFea)
        for j in range(numOfFea):
            population[i][j] = V[j]
    return population

def checkTerminationStatus(Times, oldFitness, minimumFitness):
    if Times >= 30:
        print "***** No need to continue. The fitness not changed in the last 30 generation"
        i = raw_input()
        exit(0)
    elif minimumFitness == oldFitness:
        Times += 1
    elif minimumFitness < oldFitness:
        oldFitness = minimumFitness
        Times = 0
        print "\n***** time is = ", time.strftime("%H:%M:%S", time.localtime())
        print "******************** Times is set back to 0 ********************\n"
    return oldFitness, Times


def mlrbpso_newpop(localBestMatrix, velocity, population, 
                    globalBestRow, numOfGen):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    alpha = 0.5 - ((0.5-0.33)/numOfGen)
    pval = (0.5) * (1 + alpha)
    new_pop = random.random((numOfPop, numOfFea))
    for j in range(numOfFea):
        for i in range(numOfPop):
            if (velocity[i][j] <= alpha):
                new_pop[i][j] = population[i][j]
            elif (velocity[i][j] > alpha and velocity[i][j] <= pval):
                new_pop[i][j] = localBestMatrix[i][j]
            elif (velocity[i][j] > pval and velocity[i][j] <= 1):
                new_pop[i][j] = globalBestRow[j]
            else:
                new_pop[i][j] = population[i][j]
    return new_pop

def mlrbpso_newLocalBest(localBestMatrix, localBestMatrix_fitness,
                            population, fitness):
    lowest = 1000
    numOfPop = fitness.shape[0]
    for i in range(numOfPop):
        if (fitness[i] < localBestMatrix_fitness[i]):
            localBestMatrix[i] = population[i]
            localBestMatrix_fitness[i] = fitness[i]
        if lowest > fitness[i]:
            lowest = fitness[i]

    print "Lowest fitness this generation: %s" % lowest
    return localBestMatrix, localBestMatrix_fitness

def mlrbpso_newGlobalBest(localBestMatrix, localBestMatrix_fitness, 
                            globalBestRow, globalBestRow_fitness, generations):
    elite1, elite1Index = findFirstElite(localBestMatrix_fitness, localBestMatrix)
    
    if (localBestMatrix_fitness[elite1Index] > globalBestRow_fitness):
        return globalBestRow, globalBestRow_fitness, generations
    else:
        if localBestMatrix_fitness[elite1Index] != globalBestRow_fitness:
            print "New global best: %s\n" % localBestMatrix_fitness[elite1Index]
            generations = 0
        generations += 1
        return elite1, localBestMatrix_fitness[elite1Index], generations

def mlrbpso(model, fitness, sumOfFitnesses, population,
                  elite1, elite1Index,
                  TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):

    globalBestRow = elite1
    globalBestRow_fitness = fitness[elite1Index] 
    localBestMatrix = population
    localBestMatrix_fitness = fitness
    maxGenerations = 30
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    inertiaWeight = 0.9
    velocity = random.random((numOfPop, numOfFea))
    c1 = 2
    c2 = 2
    numOfGen = 1
    unfit = 1000
    fittingStatus = 0
    generations = 0

    print "Global Best Fitness: %s\n" % globalBestRow_fitness 
    while generations < 30:
        fittingStatus, fitness = validate_model(model, population, 
                TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
        for j in range(numOfFea):
            for i in range(numOfPop):
                term1 = c1 * random.random() * (localBestMatrix[i][j] - population[i][j])
                term2 = c2 * random.random() * (globalBestRow[j] - population[i][j])
                velocity[i][j] = (inertiaWeight * velocity[i][j]) + term1 + term2
        population = mlrbpso_newpop(localBestMatrix, velocity, population,
            globalBestRow, numOfGen)
        numOfGen += 1
        localBestMatrix, localBestMatrix_fitness = mlrbpso_newLocalBest(localBestMatrix,
            localBestMatrix_fitness, population, fitness)
        globalBestRow, globalBestRow_fitness, generations = mlrbpso_newGlobalBest(localBestMatrix,
            localBestMatrix_fitness, globalBestRow, globalBestRow_fitness, generations)

    print "Could not find a better fitness after %s generations.\nFinal Global Best Fitness is %s" % (maxGenerations, globalBestRow_fitness)
    return None

def main():
    set_printoptions(threshold='nan')

    model = linear_model.LinearRegression()
    numOfPop = 200  # should be 200 population, lower is faster but less accurate
    numOfFea = 385  # should be 385 descriptors

    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = getAllOfTheData()
    TrainX, ValidateX, TestX = rescaleTheData(TrainX, ValidateX, TestX)

    unfit = 1000  # when to stop when the model isn't doing well

    fittingStatus = unfit
    while fittingStatus == unfit:
        population = createInitialPopulation(numOfPop, numOfFea)
        fittingStatus, fitness = validate_model(model, population,
             TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    elite1, elite1Index = findFirstElite(fitness, population)

    numOfPop = fitness.shape[0]
    sumFitnesses = zeros(numOfPop)
    for i in range(0, numOfPop):
        sumFitnesses[i] = fitness[i] + sumFitnesses[i - 1]

    mlrbpso(model, fitness, sumFitnesses, population,
                  elite1, elite1Index,
                  TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    return

main()



