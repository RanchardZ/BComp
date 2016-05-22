import numpy as np
from numpy.matlib import repmat
from geneticToolBox import rand_1_variation, DEExponentialCrossover, DEBinomialCrossover
from copy import deepcopy
import unittest



def DEWrapper(mc):
	def DEInner(population, values, parents, pValues, pIdxes, f, cp):
		popNum, dimNum = population.shape
		if hasattr(f, 'shape') and f.shape[0] == dimNum:
			f, cp 				= repmat(f, popNum, 1), np.ones(popNum) * cp
		else:
			f, cp 				= np.ones(popNum) * f, np.ones(popNum) * cp
		for feat, value, idx in zip(parents, pValues, pIdxes):
			neighborIdxes = range(popNum)
			del neighborIdxes[idx]
			newFeat = mc(population, values, neighborIdxes, feat, f[idx], cp[idx])
			yield (newFeat, value, idx)
	return DEInner

@DEWrapper
def rand_1_exp(population, values, neighborIdxes, feat, f, cp):
	z = rand_1_variation(population, neighborIdxes, f)
	newFeat = DEExponentialCrossover(feat, z, cp)
	return newFeat

@DEWrapper
def rand_1_bin(population, values, neighborIdxes, feat, f, cp):
	z = rand_1_variation(population, neighborIdxes, f)
	newFeat = DEBinomialCrossover(feat, z, cp)
	return newFeat

class TestDEWrapper(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.population = np.random.rand(10, 5)
		cls.originalPopulation = deepcopy(cls.population)
		cls.values = np.random.rand(10)
		cls.neighborIdxes = np.arange(10)
		cls.feat = cls.population[2]
		cls.rand_1_exp_gen = rand_1_exp(cls.population, cls.values, cls.population, cls.values, cls.neighborIdxes, 0.5, 0.5)

	@staticmethod
	def twoDimArrayEqual(arrayA, arrayB):
		assert(arrayA.shape == arrayB.shape)
		rows, cols = arrayA.shape
		for i in range(rows):
			for j in range(cols):
				if arrayA[i, j] != arrayB[i, j]:
					return False
		return True

	def testInputOutput(self):
		(sample, value, idx) = self.rand_1_exp_gen.next()

		self.assertEqual(
			type(sample),
			type(np.array([1, 2, 3]))
		)
		self.assertEqual(
			len(sample),
			5
		)

	def testDynamicChangeOfPopulation(self):
		(sample, value, idx) = self.rand_1_exp_gen.next()
		self.population[3] = sample
		self.failIf(self.twoDimArrayEqual(self.population, self.originalPopulation))


if __name__ == '__main__':
	unittest.main()