import random
import math

cityData = {}
cityDistanceLi = []

class Individual:
	def __init__(self, _size, cgenes, popInitMethod):
		"""
		Parameters and general variables
		"""
		self.fitness    = 0
		self.genes      = []
		self.genSize    = _size

		global cityData
		global cityDistanceLi

		self.cityData 		= cityData
		self.cityDistanceLi = cityDistanceLi

		if cgenes: # Child genes from crossover
			self.genes = cgenes
		else:   # Random initialisation of genes
			self.genes = list(self.cityData.keys())
			self.popInit(popInitMethod)

	def popInit(self, method):
		"""
		0		Random Tour Generation
		1		Nearest Neighbour Assignment
		"""
		if method == 0:
			random.shuffle(self.genes)
		elif method == 1:
			firstCity = random.choice(self.genes)
			# print(firstCity, end=' ')
			temp = [firstCity]
			for _ in range(len(self.genes)-1):
				distLi = cityDistanceLi[temp[-1]].copy()
				distLi = [distLi[i] if i not in temp else math.inf for i in range(len(distLi))]
				nextCity = min(range(len(distLi)), key = lambda x: distLi[x])
				temp.append(nextCity)
			self.genes = temp
		else:
			raise Exception('Wrong initialisation heuristic option')

	def copy(self):
		"""
		Creating a copy of an individual
		"""
		# ind = Individual(self.genSize, self.genes[0:self.genSize], None)
		ind = Individual(self.genSize, self.genes, None)
		ind.fitness = self.getFitness()
		return ind

	def getFitness(self):
		return self.fitness

	getDistance = lambda self, c1, c2: self.cityDistanceLi[c1][c2]

	def computeFitness(self):
		"""
		Computing the cost or fitness of the individual
		"""
		self.fitness    = self.getDistance(self.genes[0], self.genes[len(self.genes)-1])
		for i in range(0, self.genSize-1):
			self.fitness += self.getDistance(self.genes[i], self.genes[i+1])
