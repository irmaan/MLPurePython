import math
import random
import time
import numpy as np
import xml.etree.ElementTree as ET


class Ant(object):
    def __init__(self, antId):
        self.id = antId
        self.tour = []
        self.tourLength = 0

def readData(file,dimension):
    tree = ET.parse(file)
    root = tree.getroot()
    graph = root.find('graph')

    distances=np.zeros((dimension,dimension))
    i=-1
    for vertex in graph.getiterator('vertex'):
        i+=1
        for edge in vertex.getiterator('edge'):
                   distances[i][int(edge.text)]=edge.attrib['cost']

    return distances


def calculateDistance(start, end):

    return distances[start][end]

def calculateCost(antTour):
    
    antTourLength = 0
    
    for i in range(0, len(antTour)-1):
        antTourLength += calculateDistance(antTour[i], antTour[i+1])
        
    return antTourLength


def  nearestNeighbor(cities):
    # nearest neighbor for initialization pheromone
    path = []
    remove = 0
    tourLength = 0
    
    startingCity = cities[len(cities)-1]
    path.append(startingCity)
    cities.remove(startingCity)    
    
    while len(cities) > 0:
        minDistance = calculateDistance(startingCity, cities[0])
        remove = 0
        for i in range(1,len(cities)):
            
            distance = calculateDistance(startingCity, cities[i])
            if distance!=0  and  distance < minDistance:
                minDistance = distance
                nextCity = cities[i]
                remove = i
        startingCity = nextCity        
        cities.pop(remove)
        path.append(nextCity)
        tourLength += minDistance
            
    path.append(path[0])
    tourLength += calculateDistance(nextCity, path[0])
    
    print ("Nearest Neighbor tour length:", tourLength)

    return tourLength
      

def initializePheromone(cities, pheromoneMatrix, t0):

    for i in range(0,len(cities)):
        pheromoneMatrix.append([])
        for j in range(0, len(cities)):
            if i == j:
                pheromoneMatrix[i].append(0)
            else:    
                pheromoneMatrix[i].append(t0)
                       

def calculateTEtha(currentCity, cities, beta, tEtha):
    
    total = 0
    
    for i in range(0,len(cities)):
        try:
            tEthaVal = math.pow(pheromoneMatrix[currentCity][i], alpha) * math.pow(1.0/calculateDistance(currentCity, cities[i]), beta)
        except ZeroDivisionError:
            tEthaVal = 0
        tEtha.append(tEthaVal)
        total += tEthaVal
    return total


def findNextCity(currentCity, cities, beta, q0):

    rand = random.random()
    tEtha = []
    
    totalTEtha = calculateTEtha(currentCity, cities, beta, tEtha)

    if rand < q0:
        argmax = max(tEtha)
        return cities[tEtha.index(argmax)]
    
    else:
        roulette = 0
        rand = random.uniform(0, totalTEtha)
        for i in range(0,len(cities)):
            
            roulette += tEtha[i]
            if rand < roulette:
                return cities[i]
            

def tourConstruction(ant, numberOfCities, cities, beta, q0):
    currentCity = ant.tour[0]
    cities.remove(currentCity)
    
    for i in range(0,numberOfCities-2):
        
        nextCity = findNextCity(currentCity, cities, beta, q0)
        ant.tour.append(nextCity)
        currentCity = nextCity
        cities.remove(nextCity)
        
    ant.tour.append(cities.pop())
    ant.tour.append(ant.tour[0])
    
    ant.tourLength = calculateCost(ant.tour)
        
        
def localPheromoneUpdate(ant, phrMatrix, t0, ksi):
    
    for i in range(0, len(ant.tour)-1):
        current = ant.tour[i]
        next = ant.tour[i+1]

        phrMatrix[current][next] = ((1 - ksi) * phrMatrix[current][next] ) + (ksi * t0)
        phrMatrix[next][current] = phrMatrix[current][next]


def globalPheromoneUpdate(globalBestTour, globalBestTourLength, phrMatrix, rho):
    
    for i in range(0, len(globalBestTour)-1):
        current = globalBestTour[i]
        next = globalBestTour[i+1]

        phrMatrix[current][next] = ((1 - rho) * phrMatrix[current][next] ) + (Q * (1/globalBestTourLength))
        phrMatrix[next][current] = phrMatrix[current][next]


def localSearch(antId, antTour, antTourLength):
    
    
    while True:
        
        best = antTourLength
        
        for i in range(0, len(antTour)-1):
            for j in range(i+1, len(antTour)):
                newAntTour = list(antTour)
                k, l = i, j
                
                while k < l:
                    newAntTour[k], newAntTour[l] = newAntTour[l], newAntTour[k] # swap
                    
                    if k == 0:
                        newAntTour[len(antTour)-1] = newAntTour[k]
                    
                    if l == len(antTour)-1:
                        newAntTour[0] = newAntTour[l]
                    
                    k += 1
                    l -= 1
                
                newAntTourLength = calculateCost(newAntTour)
                
                if newAntTourLength < antTourLength:
                    antTourLength = newAntTourLength
                    antTour = newAntTour
                                        
                    
        if best == antTourLength:
            print (antId+1,". ant local search. Tour length:", antTourLength)
            
            return antTour, antTourLength
              


def initializeTours(bestTour, ants):
    
    del bestTour[:]
    randomCities = random.sample(range(0,numberOfCities),numberOfCities)
    random.shuffle(randomCities)
    
    for i in range(0, len(ants)):
        del ants[i].tour[:]
        ants[i].tourLength = 0
          
        ants[i].tour.append(cities[randomCities[i%10]])


def antColonySystem(iteration, cities, ants, pheromoneMatrix, numberOfCities, numberOfAnts, beta, q0, rho, ksi, t0):
    
    initializePheromone(cities, pheromoneMatrix, t0)
    
    bestTour = []
    globalBestTour = []
    
    globalBestTourLength = 0
    
    strToFile = ""
    
    for i in range (0,iteration):
        print ("\n\nIteration",i)

        bestTourLength = 0
        
        initializeTours(bestTour, ants)
        
        for j in range(0,numberOfAnts):
                        
            tourConstruction(ants[j], numberOfCities, list(cities), beta, q0)
            #localSearch()
            localPheromoneUpdate(ants[j], pheromoneMatrix, t0, ksi)
            
            print ("\n", j+1 ,". ant's tour. Tour length: ", ants[j].tourLength)

            time.sleep(0.5)
            
            ants[j].tour, ants[j].tourLength = localSearch(ants[j].id, list(ants[j].tour), ants[j].tourLength)
               
            if bestTourLength == 0 or bestTourLength > ants[j].tourLength:
                bestTourLength = ants[j].tourLength
                bestTour = ants[j].tour
                
                print (j+1,". ant,best tour. Tour length:", bestTourLength)
                  
        
        if globalBestTourLength == 0 or globalBestTourLength > bestTourLength:
            globalBestTourLength = bestTourLength
            
            print ("\nBest tour until now. Tour length: ", globalBestTourLength)
            
            globalBestTour = bestTour

        
        globalPheromoneUpdate(globalBestTour, globalBestTourLength, pheromoneMatrix, rho)
        
        
    for i in range(0, len(globalBestTour)):
       print ("\nBest tour", i+1, ". city :",globalBestTour[i])
        
    print ("\nBest Tour Length: ", globalBestTourLength)


    
def runACS(dataSet):
    global cities, numberOfCities

    numberOfCities = len(cities)
    
    t = 1/(len(cities) * nearestNeighbor(list(cities))) # copying cities list and send nn algorithm

    
    # create Ants
    for i in range(0,numberOfAnts):
        ants.append(Ant(i))
        
    antColonySystem(iteration, cities, ants, pheromoneMatrix, numberOfCities, numberOfAnts, beta, q0, rho, ksi, t)
    

numberOfCities = 535
iteration = 20
ants = []
pheromoneMatrix = []
numberOfAnts = numberOfCities

dataSet='ali535.xml'
#dataSet='bayg29.xml'

Q = 10 # amount of pheromone which an ant puts on the path

alpha = 1.0 # heuristic parameter
beta = 3.0 # heuristic parameter
q0 = 0.9 # control parameter for random proportional

rho = 0.1 # pheromone coefficient
ksi = 0.1 # local pheromone


cities = random.sample(range(0,numberOfCities),numberOfCities)

distances=readData(dataSet,numberOfCities)
runACS(dataSet)
