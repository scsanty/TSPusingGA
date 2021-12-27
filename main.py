import random
from Individual import *
import sys
import os

try:
	_graphics = True
	import matplotlib
	import matplotlib.pyplot as plt
except:
	_graphics = False

if _graphics:
	try:
		matplotlib.use('Qt5Agg')
	except:
		matplotlib.use('TkAgg')

myStudentNum = 195638 # R00195638
random.seed(myStudentNum)

class Graphics:
	def __init__(self):
		self.arrowPlots = []
		self.distLine = []
		self.distLinePoints = []
		self.cityPointPlots = []
		self.data = {}
		self.nRun = 1
		self.maxIterations = 0
		self.initH = ''
		self.plt = plt
		self.fig = self.plt.figure(num='Observer')
		self.travelPlot = self.fig.add_axes([0.02, 0.02, 0.7 ,0.7])
		self.travelPlot.axis('off')
		self.distCostPlot = self.fig.add_axes([0.7, 0.7, 0.25 ,0.25], xlabel='Iteration', ylabel='Dist Cost')
		self.distCostPlot.tick_params(which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
		self.labelAxes = self.fig.add_axes([0.1, 0.85, 0.06 ,0.06])
		self.labelAxes.axis('off')
		self.label = self.labelAxes.text(0, 0, '')

	def redraw(self):
		'''Redraws the canvas for plot removal changes'''
		self.fig.canvas.draw_idle()

	def init_plot(self):
		'''Initialise the plot with the City points on map'''
		for x, y in self.data.values():
			self.cityPointPlots.append(self.travelPlot.scatter(x, y))
		self.plt.pause(0.0000000001)

	def stay_on(self):
		'''Leave the plots on'''
		self.plt.show()

	twoPoints = lambda self, points: [points[i:i+2] for i in range(len(points)-1)]
	# Two adjacents points in a route to draw arrows
	def plotTravel(self, route, itrNo_distCost, label=''):
		self.plotLabel(f'{self.initH}   nRun: {self.nRun}\n{label}')

		if itrNo_distCost:
			while self.distLine:
				# Remove line plots
				self.distLine.pop().remove()

			i, d = itrNo_distCost
			# iteration, distance cost
			self.distLinePoints.append((i+self.nRun*self.maxIterations, d))
			# For a continuous plot over multiple iterations
			x_li, y_li = zip(*self.distLinePoints)
			# Getting lists for plots
			self.distLine = self.distCostPlot.plot(x_li, y_li, color='blue')

		while self.arrowPlots:
			# Remove arrows from plots
			self.arrowPlots.pop().remove()

		route = [self.data[city] for city in route]
		# Converting city names in route to coordinates
		route.append(route[0])
		# Appending the first city to complete the loop

		# Visualise tour
		# Based on https://gist.github.com/payoung/6087046
		for (x1, y1), (x2, y2) in self.twoPoints(route):
			# Getting two adjacents points to build arrow markers
			self.arrowPlots.append(self.travelPlot.arrow(x1, y1, (x2-x1), (y2-y1), head_width=x1/100, color='gray', length_includes_head=True))

		self.plt.pause(2)

	def plotLabel(self, text):
		self.label.set_text(text)

	def plot(self, x_li, y_li):
		'''To plot a separate temp figure'''
		fig = self.plt.figure(num='Comparisons')
		comp = fig.add_axes([0.1, 0.1, 0.8 ,0.8], xlabel='nRun', ylabel='Dist Cost', xticks=list(range(1, len(x_li)+1)))
		comp.plot(x_li, y_li, color='blue')
		# plotting Comparisons between nRuns


class BasicTSP:
	def __init__(self, _fName, _maxIterations, _popSize, _initPop, _xoverProb, _mutationRate, _trunk,  _elite):
		"""
		Parameters and general variables
		Note not all parameters are currently used, it is up to you to implement how you wish to use them and where
		"""

		self.population     = []
		self.sorted_pop 	= []
		self.matingPool     = []
		self.best           = None
		self.popSize        = int(_popSize)
		self.genSize        = None
		self.crossoverProb  = float(_xoverProb)
		self.mutationRate   = float(_mutationRate)
		self.maxIterations  = int(_maxIterations)
		self.trunkSize      = float(_trunk)
		self.eliteSize      = float(_elite)
		self.elites			= []
		self.fName          = _fName
		self.initHeuristic  = int(_initPop)
		self.iteration      = 0
		global cityData
		self.data           = cityData
		self.graphics		= None

		self.readInstance()
		self.populateDistanceList()
		self.initPopulation()



	def add_graphics(self, graphics):
		'''To add a graphical observer'''
		self.graphics = graphics
		graphics.data = self.data
		# Sending data information to Graphics class for plots
		graphics.maxIterations = self.maxIterations
		# Sending max iteration information to Graphics class for title
		graphics.initH = "Random H" if self.initHeuristic == 0 else "Nearest Neighbour H"
		# Sending Heiristic name information to Graphics class for title
		graphics.init_plot()
		# Plotting the canvas with city coordinates
		self.initPopulation(plotOnly=True)
		# Plotting the initial route between cities

	def readInstance(self):
		"""
		Reading an instance from fName
		"""
		file = open(self.fName, 'r')
		self.genSize = int(file.readline())
		for line in file:
			(cid, x, y) = line.split()
			self.data[int(cid)] = (int(x), int(y))
		file.close()

	def populateDistanceList(self):
		'''To create a pre-computed distance 2D List'''
		global cityDistanceLi

		for i in range(len(cityData)+1):
			innerLi = []
			for j in range(len(cityData)+1):
				if 0 in {i, j} or j == i:
					# Fills math.inf if the city number is 0 or is equal
					# math.inf works well with min() function, otherwise None
					# could have been used
					innerLi.append(math.inf)
				else:
					# Computing distance for the rest of the cities
					x1, y1 = cityData[i]
					x2, y2 = cityData[j]
					dist = math.sqrt((x1-x2)**2 + (y1-y2)**2 )
					innerLi.append(dist)
			cityDistanceLi.append(innerLi)

	def initPopulation(self, plotOnly=False):
		"""
		Creating random individuals in the population
		"""
		# Following if is to insinuate Method Overloading
		if not plotOnly:
			# Empty Population
			self.population = []
			for i in range(0, self.popSize):
				# Create popSize number of Individual
				individual = Individual(self.genSize, [], self.initHeuristic)
				individual.computeFitness()
				self.population.append(individual)

			self.sorted_pop = sorted(self.population, key=lambda x: x.getFitness())
			# Sort the population by its fitness to be used for trucation and elitism

			self.best = self.population[0].copy()
			# Set a best Solution
			for ind_i in self.population:
				# Compare the best solution with every candidate and udpate best
				if self.best.getFitness() > ind_i.getFitness():
					self.best = ind_i.copy()
			print ("Best initial sol: ",self.best.getFitness())

		if self.graphics:
			# Plotting the initial findings
			self.graphics.plotTravel(self.best.genes, (0, self.best.getFitness()),
				"Best initial sol: {self.best.getFitness()}")

	def updateBest(self, candidate):
		'''Updating the Best candidate after every new generation'''
		if self.best == None or candidate.getFitness() < self.best.getFitness():
			# Lesser fitness is better because lesser fitness is lesser distance
			self.best = candidate.copy()
			print ("iteration: ", self.iteration, "best: ",self.best.getFitness())

			if self.graphics:
				# Plotting the best into the graph
				self.graphics.plotTravel(self.best.genes, (self.iteration, self.best.getFitness()),
					f"iteration: {self.iteration} best: {round(self.best.getFitness(), 2)}")

	def randomSelection(self):
		"""
		Random (uniform) selection of two individuals
		"""
		indA, indB = random.sample(self.matingPool, k=2)
		return [indA, indB]

	def truncationTournamentSelection(self):
		"""
		Your Truncation Tournament Selection Implementation to fill the mating pool
		"""
		if float(self.trunkSize) == 0.0:
			# In case there is 0 truncation, full population is to be used as
			# as mating pool
			self.matingPool = self.population
		else:
			cutoffIndx = int(self.popSize * self.trunkSize)
			# Get cutoff based on the percentage
			self.matingPool = self.sorted_pop[0:cutoffIndx]
			# Truncate the sorted population till cut off

	def updateElites(self):
		if float(self.eliteSize) == 0.0:
			return
		cutoffIndx = int(self.popSize * self.eliteSize)
		# Get cutoff based on the percentage
		self.elites = self.sorted_pop[0:cutoffIndx]
		# Slice the sorted population till cut off

	def order1Crossover(self, indA, indB):
		"""
		Your Order1 Crossover Implementation
		"""
		if random.random() <= self.crossoverProb:
			indx1, indx2 = sorted(random.sample(range(self.genSize), k=2))
			# Getting two random cutoff indices
			child = [None] * self.genSize
			# Creating an empty gene sequence i.e. empty list
			child[indx1:indx2] = indA.genes[indx1:indx2]
			# inserting Individual A's genes in between the cutoff points
			indB = [allele for allele in indB.genes if allele not in child]
			# Removing above alleles from Individual B's gene sequence
			child[0:indx1], child[indx2:] = indB[0:indx1], indB[indx1:]
			# inserting the rest of the genes in Individual B into child
			child = Individual(self.genSize, child, None)
			# Creating an individual with crossover genes
		else:
			# If the probability didn't satisfy
			# send a parent at random to become as child for next generation
			child = random.choice([indA, indB])

		return child

	def inversionMutation(self, ind):
		"""
		Your Inversion Mutation implementation
		"""
		if random.random() < self.mutationRate:
			indx1, indx2 = sorted(random.sample(range(self.genSize), k=2))
			# Get two indices
			ind.genes[indx1:indx2] = (ind.genes[indx1:indx2])[::-1]
			# Reverse them using list slicing operator [first:last:step]
			# directly into the same location as the cutoff points

	def newGeneration(self):
		"""
		Creating a new generation
		1. Selection
		2. Crossover
		3. Mutation
		"""
		for i in range(self.popSize):
			"""
			Depending of your experiment you need to use the most suitable algorithms for:
			1. Select two candidates
			2. Apply Crossover
			3. Apply Mutation
			"""
			if self.population[i] in self.elites:
				# If the parent is in elites, do not participate in the following
				continue
			parent1, parent2 = self.randomSelection()
			# Choose two individuals from the mating pool
			child = self.order1Crossover(parent1,parent2)
			# Perform crossover, depending on the crossover probaility
			self.inversionMutation(child)
			# Perform mutation, depending on the mutation rate
			child.computeFitness()
			# Perform total distance calculation, i.e. fitness
			self.updateBest(child)
			# Peform best updation
			self.population[i]=child
			# Update the population

		self.sorted_pop = sorted(self.population, key=lambda x: x.getFitness())
		# Sort the population by its fitness to be used for trucation and elitism


	def GAStep(self):
		"""
		One step in the GA main algorithm
		1. Updating mating pool with current population
		2. Creating a new Generation
		"""
		self.truncationTournamentSelection()
		# Update mating Pool
		self.updateElites()
		# Update elite's li
		self.newGeneration()
		# Update population with new generation

	def search(self):
		"""
		General search template.
		Iterates for a given number of steps
		"""
		self.iteration = 0
		while self.iteration < self.maxIterations:
			self.GAStep()
			self.iteration += 1
			print(f"{self.iteration}/{self.maxIterations}", end='\r')

		print ("Total iterations: ", self.iteration)
		print ("Best Solution: ", self.best.getFitness())
		if self.graphics:
			# Plot the final best solution
			self.graphics.plotTravel(self.best.genes, (self.iteration, self.best.getFitness()),
				f"Total iterations: {self.iteration} Best: {round(self.best.getFitness(), 2)}")


def sanityChecks(inst, nRuns, nIters, pop, initH, pC, pM, trunkP, eliteP):
	'''To check if the input variables are correct'''
	# String to correct datatypes
	nRuns 	= int(nRuns)
	nIters 	= int(nIters)
	pop 	= int(pop)
	initH 	= int(initH)
	pC 		= float(pC)
	pM 		= float(pM)
	trunkP 	= float(trunkP)
	eliteP 	= float(eliteP)

	if not os.path.exists(inst):
		raise Exception("File path is invalid")

	if nRuns < 1 or nIters < 1:
		raise Exception("Max iterations/Number of full iterations cannot be negative for a fraction")

	if pop < 1:
		raise Exception("Population size cannot be negative for a fraction")

	if initH not in [0, 1]:
		raise Exception('Enter the vaule of initialisation heuristic as 0 for Random tour, 1 for Nearest neighbour')

	if list(filter(lambda x: not (0 <= x <= 1), [trunkP, pM, eliteP, pC])):
		raise Exception("Please enter a value between 0 and 1 for crossover prob, mutation rate, trunk %, elite % (fractional percentages)")

	if eliteP >= trunkP:
		raise Exception("Elitism % greater than or equal Truncation % will create no children")

	return inst, nRuns, nIters, pop, initH, pC, pM, trunkP, eliteP

if __name__ == '__main__':
	if len(sys.argv) < 10:
		print ("Error - Incorrect input")
		print ("Expecting python TSP.py [instance] [number of runs] [max iterations] [population size]",
				"[initialisation method] [xover prob] [mutate prob] [truncation] [elitism]")
		sys.exit(0)

	Individual.cityData = {}
	Individual.cityDistanceLi = []

	_, inst, nRuns, nIters, pop, initH, pC, pM, trunkP, eliteP = sys.argv
	# Reading in parameters

	inst, nRuns, nIters, pop, initH, pC, pM, trunkP, eliteP = sanityChecks(inst, nRuns, nIters, pop, initH, pC, pM, trunkP, eliteP)
	# inst, nRuns, nIters, pop, initH, pC, pM, trunkP, eliteP = r"./TSP_dataset/inst-19.tsp", 10, 500, 100, 1, 0.8, 0.05, 0.25, 0.0


	ga = BasicTSP(inst, nIters, pop, initH, pC, pM, trunkP, eliteP)
	nRun_best = []
	# Similar functions, with and without graphical representation
	if _graphics:
		graphics = Graphics()
		ga.add_graphics(graphics)

	ga.search()
	nRun_best=[(1, ga.best.getFitness(), ga.best.genes)]
	for i in range(1, nRuns):
		if _graphics:
			graphics.nRun = i+1
		ga.initPopulation()
		ga.search()
		nRun_best.append((i+1, ga.best.getFitness(), ga.best.genes))

	nRun_li, best_li, gene_li = zip(*nRun_best)
	best_distance = min(best_li)
	best_indx = best_li.index(best_distance)
	best_route = gene_li[best_indx]
	# Getting the best route out of all the iterations
	best_nRun = nRun_li[best_indx]

	if _graphics:
		# Plotting the best into the graph
		graphics.nRun = best_nRun
		graphics.plotTravel(best_route, None, f"Total iterations: {(nRuns*nIters)} Best: {round(best_distance, 2)}")
		graphics.plot(nRun_li, best_li)
		graphics.stay_on()

	best_route.append(best_route[0])
	with open('Final_route.txt', 'w') as f:
		# Saving the best reoute into a file
		f.write(f'{best_distance}\n{best_route}')
