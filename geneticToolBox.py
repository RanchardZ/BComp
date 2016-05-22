import numpy as np
from copy import copy

import unittest
# ---------- #
#  variator  #
# ---------- #
def variator(variate):
	def DEVariator(population, neighborIdxes, f):
		variant = variate(population, neighborIdxes, f)
		return variant
	return DEVariator

@variator
def rand_1_variation(population, neighborIdxes, f):
	a, b, c = population[np.random.choice(neighborIdxes, 3, replace=False)]
	variant = a + f * (b - c)
	return variant

# ----------- #
#  crossover  #
# ----------- #
def DECrossover(cross):
	def crossover(feat, z, cp):
		newFeat = cross(feat, z, cp)
		return newFeat
	return crossover

@DECrossover
def DEExponentialCrossover(feat, z, cp):
	dimNum 	= len(feat)
	newFeat 	= copy(feat)
	# make sure that the offspring is different from its parent
	di 				= np.random.randint(dimNum)
	newFeat[di] 	= z[di]

	pointer = np.random.randint(dimNum)
	count 	= 0
	while (np.random.rand() < cp) and (count < dimNum):
		di 						= (pointer + count) % dimNum
		newFeat[di]			= z[di]
		count 					+= 1
	return newFeat

@DECrossover
def DEBinomialCrossover(feat, z, cp):
	dimNum = len(feat)
	r = np.random.rand(dimNum)
	newFeat = feat * (r > cp) + z * (r <= cp)
	return newFeat
# ------------- #
#  replacement  #
# ------------- #
def plusReplacement(population, values, children, cValues):
	popNum, dimNum = population.shape

	pool = np.vstack((population, children))
	pValues = np.hstack((values, cValues))

	sortedIdxes = np.argsort(pValues)
	return pool[sortedIdxes[: popNum]], pValues[sortedIdxes[: popNum]]

# ----------- #
#  selection  #
# ----------- #
def uniformSelection(population, values, num):
	""" randomly select individuals from population as parents (There is no competition)
	"""
	popNum, dimNum 	= population.shape
	chosenIdxes 	= np.random.choice(np.arange(popNum), num, replace = False)
	return population[chosenIdxes], values[chosenIdxes], chosenIdxes


class TestVariator(unittest.TestCase):

	def test_rand_1_variation(self):
		population = np.array([
			[1, 2, 3, 4],
			[1, 2, 3, 4],
			[1, 2, 3, 4],
			[4, 5, 6, 7]
		])
		neighborIdxes = np.array([0, 1, 2])
		actual = rand_1_variation(population, neighborIdxes, 0.5)
		expected = np.array([1, 2, 3, 4])
		for act, exp in zip(actual, expected):
			self.assertEqual(act, exp)

class TestCrossover(unittest.TestCase):

	def testDEExponentialCrossover(self):
		feat = np.array([1, 2, 3])
		z = np.array([4, 5, 6])
		self.assertEqual(
			type(DEExponentialCrossover(feat, z, 0.5)),
			type(feat)
		)

class TestReplacement(unittest.TestCase):

	def testPlusReplacement(self):
		population = np.arange(20).reshape(4, 5)
		values = np.array([1, 2, 3, 4])
		children = np.arange(40)[20:].reshape(4, 5)
		cValues = np.array([0, 5, 6, 7])

		rPopulation, rValues = plusReplacement(population, values, children, cValues)
		self.assertEqual(rValues[0], 0)
		self.assertEqual(rValues[1], 1)

class TestSelection(unittest.TestCase):

	def testUniformSelection(self):
		population = np.random.rand(5, 10)
		values = np.random.rand(5)

		sPopulation, sValues, sIdxes = uniformSelection(population, values, 1)
		self.failUnless(sValues[0] in values)

if __name__ == '__main__':
	unittest.main()
