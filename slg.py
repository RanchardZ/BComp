import numpy as np
from copy import copy		

class SLG(object):
	def __init__(self, popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations):
		self.popNum = popNum
		self.dimNum = dimNum
		self.problemID = problemID
		self.budget = budget

		self.safeEvaluate = safeEvaluate
		self.safeGetEvaluations = safeGetEvaluations

		self.initPop()

		self.lp = 0.5 # local search probability

	def initPop(self):
		self.population = np.random.rand(self.popNum, self.dimNum)
		self.values = np.zeros(self.popNum)
		for i, individual in enumerate(self.population):
			self.values[i] = self.safeEvaluate(self.problemID, individual)
			if self.safeGetEvaluations(self.problemID) >= self.budget:
				break

	def optimize(self):
		while self.safeGetEvaluations(self.problemID) < self.budget:
			param2change = np.random.randint(0, self.dimNum, self.popNum)
			fitnesses = getFitnesses(self.values)
			probs = getProbs(fitnesses)

			for _ in range(self.popNum):
				if np.random.rand() < self.lp:
					selectIndex = rouletteWheelSelection(probs)
				else:
					selectIndex = np.random.randint(0, self.popNum)
				neighborIndex = np.random.randint(0, self.popNum)
				while neighborIndex == selectIndex:
					neighborIndex = np.random.randint(0, self.popNum)
				selectedInd = self.population[selectIndex]
				neighborInd = self.population[neighborIndex]
				newInd = copy(selectedInd)
				paramIndex = param2change[selectIndex]
				newInd[paramIndex] = selectedInd[paramIndex] + \
									(selectedInd[paramIndex] - neighborInd[paramIndex]) * \
									(np.random.rand() - 0.5) * 2
				newInd = self.bounder(newInd)
				newValue = self.safeEvaluate(self.problemID, newInd)
				if newValue < self.values[selectIndex]:
					self.population[selectIndex] 	= newInd
					self.values[selectIndex] 		= newValue

				if self.safeGetEvaluations(self.problemID) >= self.budget:
					break
	@staticmethod
	def bounder(ind):
		# reinitialize
		newInd = copy(ind)
		for i, d in enumerate(ind):
			if (d > 1.0 or d < 0.0):
				newInd[i] = np.random.rand()
		return newInd



def getFitnesses(values):
	pos_mask = values >= 0
	neg_mask = values < 0
	return 1. * pos_mask  / (values + 1.) +\
		  (1. - values) * neg_mask

def getProbs(fitnesses):
	m = np.max(fitnesses)
	return .9 * fitnesses / m + 0.1

def rouletteWheelSelection(areas):
	r = np.random.rand() * np.sum(areas)
	selectIndex = 0
	select = areas[selectIndex]
	while (select < r) and (selectIndex < len(areas)):
		selectIndex += 1
		select 		+= areas[selectIndex]
	return selectIndex

