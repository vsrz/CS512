import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions
from sklearn import svm  # provides Support Vector Regression
from sklearn import linear_model
import csv
import math
import sys
import hashlib



# ------------------------------------------------------------------------------


def r2(y, yHat):
    """Coefficient of determination"""
    numer = ((y - yHat) ** 2).sum()  # Residual Sum of Squares
    denom = ((y - y.mean()) ** 2).sum()
    r2 = 1 - numer / denom
    return r2


#------------------------------------------------------------------------------

def r2Pred(yTrain, yTest, yHatTest):
    numer = ((yHatTest - yTest) ** 2).sum()
    denom = ((yTest - yTrain.mean()) ** 2).sum()
    r2Pred = 1 - numer / denom
    return r2Pred


#------------------------------------------------------------------------------

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


#------------------------------------------------------------------------------


def ccc(y, yHat):
    """Concordance Correlation Coefficient"""
    n = y.shape[0]
    numer = 2 * (((y - y.mean()) * (yHat - yHat.mean())).sum())
    denom = ((y - y.mean()) ** 2).sum() + ((yHat - yHat.mean()) ** 2).sum() + n * ((y.mean() - yHat.mean()) ** 2)
    ccc = numer / denom
    return ccc


#------------------------------------------------------------------------------

def ccc_adj(ccc, n, p):
    """
    Adjusted CCC
    Parameters
    ----------
    n : int -- Sample size
    p : int -- Number of parameters

    """
    ccc_adj = ((n - 1) * ccc - p) / (n - p - 1)
    return ccc_adj


#------------------------------------------------------------------------------

def q2F3(yTrain, yTest, yHatTest):
    numer = (((yTest - yHatTest) ** 2).sum()) / yTest.shape[0]
    denom = (((yTrain - yTrain.mean()) ** 2).sum()) / yTrain.shape[0]
    q2F3 = 1 - numer / denom
    return q2F3


#------------------------------------------------------------------------------

def k(y, yHat):
    """Compute slopes"""
    k = ((y * yHat).sum()) / ((yHat ** 2).sum())
    kP = ((y * yHat).sum()) / ((y ** 2).sum())
    return k, kP


#------------------------------------------------------------------------------

def r0(y, yHat, k, kP):
    """
    Compute correlation for regression lines through the origin
    Parameters
    ----------
    k  : float -- Slope
    kP : float -- Slope

    """
    numer = ((yHat - k * yHat) ** 2).sum()
    denom = ((yHat - yHat.mean()) ** 2).sum()
    r2_0 = 1 - numer / denom
    numer = ((y - kP * y) ** 2).sum()
    denom = ((y - y.mean()) ** 2).sum()
    rP2_0 = 1 - numer / denom
    return r2_0, rP2_0


#------------------------------------------------------------------------------

def r2m(r2, r20):
    """Roy Validation Metrics"""
    r2m = r2 * (1 - (r2 - r20) ** 0.5)
    return r2m


#------------------------------------------------------------------------------

def r2m_adj(r2m, n, p):
    """
    Adjusted r2m
    Parameters
    ----------
    n : int -- Number of observations
    p : int -- Number of predictor variables

    """
    r2m_adj = ((n - 1) * r2m - p) / (n - p - 1)
    return r2m_adj


#------------------------------------------------------------------------------

def r2p(r2, r2r):
    """
    Parameters
    ----------
    r2r : float --Average r^2 of y-randomized models.

    """
    r2p = r2 * ((r2 - r2r) ** 0.5)
    return r2p


#------------------------------------------------------------------------------

def rSquared(y, yPred):
    """Find the coefficient  of correlation for an actual and predicted set.

    Parameters
    ----------
    y : 1D array -- Actual values.
    yPred : 1D array -- Predicted values.

    Returns
    -------
    out : float -- Coefficient  of correlation.

    """

    rss = ((y - yPred) ** 2).sum()  # Residual Sum of Squares
    sst = ((y - y.mean()) ** 2).sum()  # Total Sum of Squares
    r2 = 1 - (rss / sst)

    return r2


#------------------------------------------------------------------------------

def rmse(X, Y):
    """
    Calculate the root-mean-square error (RMSE) also known as root mean
    square deviation (RMSD).

    Parameters
    ----------
    X : array_like -- Assumed to be 1D.
    Y : array_like -- Assumed to be the same shape as X.

    Returns
    -------
    out : float64
    """

    X = asarray(X, dtype=float64)
    Y = asarray(Y, dtype=float64)

    return (sum((X - Y) ** 2) / len(X)) ** .5


#------------------------------------------------------------------------------

def cv_predict(set_x, set_y, model):
    """Predict using cross validation."""
    yhat = empty_like(set_y)
    for idx in range(0, yhat.shape[0]):
        train_x = delete(set_x, idx, axis=0)
        train_y = delete(set_y, idx, axis=0)
        model = model.fit(train_x, train_y)
        yhat[idx] = model.predict(set_x[idx])
    return yhat


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 16, 2013

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


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 16, 2013
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


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 16, 2013
def initializeYDimension():
    yTrain = {}
    yHatTrain = {}
    yHatCV = {}
    yValidation = {}
    yHatValidation = {}
    yTest = {}
    yHatTest = {}
    return yTrain, yHatTrain, yHatCV, yValidation, yHatValidation, yTest, yHatTest


#------------------------------------------------------------------------------
def OnlySelectTheOnesColumns(popI):
    numOfFea = popI.shape[0]
    xi = zeros(numOfFea)
    for j in range(numOfFea):
        xi[j] = popI[j]

    xi = xi.nonzero()[0]
    xi = xi.tolist()
    return xi


#------------------------------------------------------------------------------

#Ahmad Hadaegh: Modified  on: July 16, 2013
def validate_model(model, fileW, population, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
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

        # Compute R2 statistics (Prediction for Valiation and Test set)
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


    #printing the information into the file
    write(model, fileW, trackDesc, trackIdx, trackFitness, trackModel, trackR2, \
          trackQ2, trackR2PredValidation, trackR2PredTest, trackSEETrain, \
          trackSDEPValidation, trackSDEPTest, yTrain, yHatTrain, yHatCV, \
          yValidation, yHatValidation, yTest, yHatTest)

    return itFits, fitness


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 16, 2013

def write(model, fileW, trackDesc, trackIdx, trackFitness, trackModel, trackR2, \
          trackQ2, trackR2PredValidation, trackR2PredTest, trackSEETrain, \
          trackSDEPValidation, trackSDEPTest, yTrain, yHatTrain, yHatCV, \
          yValidation, yHatValidation, yTest, yHatTest):
    for key in trackFitness.keys():
        if fileW != '':
            fileW.writerow([trackDesc[key], trackIdx[key], trackFitness[key], trackModel[key], \
                trackR2[key], trackQ2[key], trackR2PredValidation[key], trackR2PredTest[key], \
                trackSEETrain[key], trackSDEPValidation[key], trackSDEPTest[key], \
                yTrain[key], yHatTrain[key], yHatCV[key], yValidation[key], yHatValidation[key], \
                yTest[key], yHatTest[key]])
        #fileOut.close()

#------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
def getTwoDecPoint(x):
    return float("%.2f" % x)


#------------------------------------------------------------------------------
def placeDataIntoArray(fileName):
    with open(fileName, mode='rbU') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar=' ')
        dataArray = array([row for row in datareader], dtype=float64, order='C')

    if (min(dataArray.shape) == 1):  # flatten arrays of one row or column
        return dataArray.flatten(order='C')
    else:
        return dataArray


#------------------------------------------------------------------------------
def getAllOfTheData():
    TrainX = placeDataIntoArray('Train-Data.csv')
    TrainY = placeDataIntoArray('Train-pIC50.csv')
    ValidateX = placeDataIntoArray('Validation-Data.csv')
    ValidateY = placeDataIntoArray('Validation-pIC50.csv')
    TestX = placeDataIntoArray('Test-Data.csv')
    TestY = placeDataIntoArray('Test-pIC50.csv')
    return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY


#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------


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
    for i in range(0, numOfPop):
        sumFitnesses[i] = fitness[i] + sumFitnesses[i - 1]

    return sumFitnesses


#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013

def OnePointCrossOver(mom, dad):
    numOfFea = mom.shape[0]
    p = int(random.uniform(0, numOfFea - 1))
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
        if (p < 0.005):
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
        print("This is generation %d --- Minimum of fitness is: -> %2f" % (i, fitness.min()))
        if (fitness.min() < 0.005):
            print "***********************************"
            print "Good: Fitness is low enough to quit"
            print "***********************************"
            exit(0)
        fittingStatus = unfit
        while (fittingStatus == unfit):
            population = findNewPopulation(elite1, elite2, sumOfFitnesses, population)
            population = createInitialPopulation(numOfPop, numOfFea)
            fittingStatus, fitness = validate_model(model, fileW,
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
    set_printoptions(threshold='nan')

    fileW = createAnOutputFile()
    fileW = ''
    #model = linear_model.LinearRegression()
    model = svm.SVR()
    numOfPop = 200  # should be 200 population, lower is faster but less accurate
    numOfFea = 385  # should be 385 descriptors

    # Final model requirements
    R2req_train = .8
    R2req_validate = .5
    R2req_test = .5

    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = getAllOfTheData()
    TrainX, ValidateX, TestX = rescaleTheData(TrainX, ValidateX, TestX)

    #numOfFea = TrainX.shape[1]  # should be 396 descriptors

    unfit = 1000  # when to stop when the model isn't doing well

    fittingStatus = unfit
    while (fittingStatus == unfit):
        population = createInitialPopulation(numOfPop, numOfFea)
        fittingStatus, fitness = validate_model(model, fileW, population,
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



