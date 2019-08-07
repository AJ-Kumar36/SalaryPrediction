import findspark
findspark.init()

from numpy import array
from pyspark.context import SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint

#set up sparkcontext
sc = SparkContext("local", "NBA Salary Predictor")

#Import Data from CSV files 
salaries = sc.textFile("C:/Users/AJ Kumar/Desktop/AJ/Projects/Practice_Py/NBA_season1718_salary.csv")
player_stats = sc.textFile("C:/Users/AJ Kumar/Desktop/AJ/Projects/Practice_Py/Seasons_Stats.csv")
print("Data Loaded")


#Filter the header from the salary csv file
salary_header = salaries.first()
raw_salaries = salaries.filter(lambda x: x != salary_header) 

#Filter the header from the stats csv file
stats_header = player_stats.first()
player_stats = player_stats.filter(lambda x: x != stats_header)

#splits the csv into individual cells and maps it to an RDD
csvSalaries = raw_salaries.map(lambda x: x.split(","))
csvStats = player_stats.map(lambda x: x.split(","))

#Filters out any stats the are not from the 2017-2018 year
filteredStats = csvStats.filter(lambda x: x[1] == "2017")
print("Filtered Stats")


#This only used to test using a small sample of the RDD rather than all of the data
#smallDataSet = csvSalaries.top(30)

def createLabeledPoint(salaries, stats):
    #name = salaries[1]
    salary = int(salaries[3])
    pos = numeratePosition(stats[3])
    age = int(stats[4])
    gamesPlayed = int(stats[6])
    gs = int(stats[7])
    mp = int(stats[8])
    per = float(stats[9])
    ts = float(stats[10])
    threeAR = float(stats[11])
    ftr = float(stats[12])
    orbPerc = float(stats[13])
    drbPerc = float(stats[14])
    trbPerc = float(stats[15])
    astPerc = float(stats[16])
    stlPerc = float(stats[17])
    blkPerc = float(stats[18])
    tovPerc = float(stats[19])
    usgPerc = float(stats[20])
    ows = float(stats[22])
    dws = float(stats[23])
    ws = float(stats[24])
    ws48 = float(stats[25])
    obpm = float(stats[27])
    dbpm = float(stats[28])
    bpm = float(stats[29])
    vorp = float(stats[30])
    fg = int(stats[31])
    fga = int(stats[32])
    fgPerc = float(stats[33])
    threeP = int(stats[34])
    threePA = int(stats[35])
    threePPerc = float(stats[36])
    twoP = int(stats[37])
    twoPA = int(stats[38])
    twoPPerc = float(stats[39])
    efg = float(stats[40])
    ft = int(stats[41])
    fta = int(stats[42])
    ftPerc = float(stats[43])
    orb = int(stats[44])
    drb = int(stats[45])
    trb = int(stats[46])
    ast = int(stats[47])
    stl = int(stats[48])
    blk = int(stats[49])
    tov = int(stats[50])
    pf = int(stats[51])
    pts = int(stats[52])
    return LabeledPoint(salary, array([pos, age, gamesPlayed, gs, mp, per, ts, threeAR, ftr, orbPerc, drbPerc, trbPerc, astPerc, stlPerc, blkPerc, \
        tovPerc, usgPerc, ows, dws, ws, ws48, obpm, dbpm, bpm, vorp, fg, fga, fgPerc, threeP, threePA, threePPerc, twoP, twoPA, twoPPerc, efg, ft, fta, ftPerc, \
            orb, drb, trb, ast, stl, blk, tov, pf, pts]))



#Converts the player's position to a number
def numeratePosition(position):    
    if position == "PG":
        return 1
    elif position == "SG":
        return 2
    elif position == "SF":
        return 3
    elif position == "PF":
        return 4
    elif position == "C":
        return 5
    else:
        return 0


labeledArray = []
#Go through all of salaries listed and match the stats of the player to his salary
#Then, use those two data points to create a LabeledPoint and append it to the array
for item in csvSalaries.collect():
    matched = filteredStats.filter(lambda x: x[2] == item[1])
    if matched.isEmpty():
        continue
    #print(item[1])
    #print(matched.first())
    labeledArray.append(createLabeledPoint(item, matched.first()))
print("Appended DataSets")


#Convert the array of labelled points into an RDD
dataSet = sc.parallelize(labeledArray)

#Split the data into testing and training
(trainingData, testData) = dataSet.randomSplit([.7, .3])

#Create the model using the training data and generate predictions using the test data
model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={}, numTrees=5, impurity='variance', maxDepth=4, maxBins=32)
predictions = model.predict(testData.map(lambda x: x.features))


results = predictions.collect()
testDataList = testData.collect()

#iterator for the test data array
count = 0

for item in results:
    #Retrieve the actual salary from the labeledpoint
    salaryMatch = testDataList[count].label
    #Find the player who has this same salary
    playerMatch = csvSalaries.filter(lambda x: float(x[3])==salaryMatch).first()
    print(playerMatch[1])
    
    #Print the salary from the csv file
    print("Actual: ", end="")
    print(playerMatch[3])
    count+=1

    #Print the salary predicted by the model
    print("Predicted: ", end="")
    print(item)