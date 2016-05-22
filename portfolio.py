import numpy as np
from math import fmod
from copy import copy

from de import *
from geneticToolBox import *

class MultiEA(object):

	def __init__(self, popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations):
		self.popNum = popNum
		self.dimNum = dimNum
		self.problemID = problemID
		self.budget = budget

		self.safeEvaluate = safeEvaluate
		self.safeGetEvaluations = safeGetEvaluations

		self.initPop()
		self.lp = 0.5

	def initPop(self):
		self.initEAs()
		self.initParams()
		self.preOptimize()

	def optimize(self):
		while self.safeGetEvaluations(self.problemID) < self.budget:
			if self.shouldPredict():
				t_nearest = np.max((self.t * np.array(self.EA2PopNum.values())).astype('int'))
				pf = np.zeros(self.NumEA)
				for i, c in enumerate(self.cc):
					sample = []
					for j in range(max(len(c) - 1 - self.length, 0), len(c) - 1):
						y = c[j:]
						x = range(len(y))
						A = np.vstack([x, np.ones(len(x))]).T
						s, t = np.linalg.lstsq(A, y)[0]
						sample.append(s * (t_nearest * 1. / self.EA2PopNum.values[i] - j) + t)

					kernal = gaussian_kde(sample)
					pf[i] = kernal.resample(1)[0, 0]
				bestEAIdx = np.argsort(pf)[0]
				self.EAs[i].optimizeOneIter()

				if len(self.cc[bestEAIdx]) >= self.length + 1:
					self.cc[bestEAIdx] = self.cc[bestEAIdx][-self.length:]
				self.cc[bestEAIdx].append(np.min(self.EAs[bestEAIdx].values))
			else:
				t_length = np.array(self.t)
				bestEAIdx = np.argsort(t_length)[-1]
				self.EAs[bestEAIdx].optimizeOneIter()

			self.t[bestEAIdx] += 1

	def initEAs(self):
		self.EAs = []
		self.EA2PopNum = {
			'CMAES': 14,
			'OBDE': 50,
			'SOUPDE': 50
		}
		self.NumEA = len(self.EA2PopNum)
		self.t = np.zeros(self.NumEA).astype('int')
		self.m = self.EA2PopNum.values()
		self.popNum = sum(self.m)
		for name, num in self.EA2PopNum.items():
			exec("EA = %s(num, self.dimNum, self.problemID, self.budget, self.safeEvaluate,self.problemID,  self.safeGetEvaluations)" % (name))
			self.EAs.append(EA)

	def initParams(self):
		self.cc = [[] for _ in range(self.NumEA)] # convergence curves
		for i in range(self.NumEA):
			self.cc[i].append(np.min(self.EAs[i].values))
		self.length = 10
		self.threshold_t = 100

	def preOptimize(self):

		for i, EA in enumerate(self.EAs):
			alpha = 1
			preBestValue = np.min(EA.values)
			EA.optimizeOneIter()
			curBestValue = np.min(EA.values)
			self.cc[i].append(curBestValue)
			while curBestValue >= preBestValue or alpha <= 2:
				preBestValue = curBestValue
				EA.optimizeOneIter()
				curBestValue = np.min(EA.values)
				self.cc[i].append(curBestValue)
				alpha += 1
			self.t[i] += alpha + 1

	def shouldPredict(self):
		return max(self.t) / min(self.t) <= self.threshold_t

class EA(object):
	def __init__(self, popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations):
		self.popNum = popNum
		self.dimNum = dimNum
		self.problemID = problemID
		self.budget = budget

		self.safeEvaluate = safeEvaluate
		self.safeGetEvaluations = safeGetEvaluations

		self.initPop()
		self.initParams()

	def optimize(self):
		while self.safeGetEvaluations(self.problemID) < self.budget:
			self.optimizeOneIter()

	def optimizeOneIter(self):
		raise NotImplementedError

	def initPop(self):
		self.population = np.random.rand(self.popNum, self.dimNum)
		self.values = np.zeros(self.popNum)
		for i, individual in enumerate(self.population):
			self.values[i] = self.safeEvaluate(self.problemID, individual)
			if self.safeGetEvaluations(self.problemID) >= self.budget:
				break

class jDE(EA):
	def __init__(self, popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations):
		super(jDE, self).__init__(popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations)

	def optimizeOneIter(self):
		parents, pValues, pIdxes = uniformSelection(self.population, self.values, self.popNum)
		maskF = np.random.rand(self.popNum) < self.alpha
		maskCP = np.random.rand(self.popNum) < self.beta
		self.params[:, 0] = (self.flb + np.random.rand(self.popNum) * self.fub) * maskF + self.params[:, 0] * (1. - maskF)
		self.params[:, 1] = np.random.rand(self.popNum) * maskCP + self.params[:, 1] * (1 - maskCP)
		# rand_1_bin
		newIndividualGen = rand_1_bin(
			self.population,
			self.values,
			parents,
			pValues,
			pIdxes,
			self.params[:, 0],
			self.params[:, 1]
		)		
		for (newIndividual, value, idx) in newIndividualGen:
			newIndividual = bounder(newIndividual)
			newValue = self.safeEvaluate(self.problemID, newIndividual)
			if newValue < value:
				self.population[idx] = newIndividual
				self.values[idx] = newValue
			if self.safeGetEvaluations(self.problemID) >= self.budget:
				return

	def initParams(self):
		self.flb, self.fub 		= 0.1, 0.9
		self.alpha, self.beta 	= 0.1, 0.1
		params 					= np.zeros(shape = (self.popNum, 2))
		params[:, 0] 			= self.flb + self.fub * np.random.rand(self.popNum)
		params[:, 1] 			= np.random.rand(self.popNum)
		self.params 			= params

class DZAdaptiveDE(EA):

	def __init__(self, popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations):
		super(DZAdaptiveDE, self).__init__(popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations)

	def optimizeOneIter(self):
		parents, pValues, pIdxes = uniformSelection(self.population, self.values, self.popNum)
		newIndividualGen = rand_1_bin(
			self.population,
			self.values,
			parents,
			pValues,
			pIdxes,
			self.F,
			self.CP
		)
		for (newIndividual, value, idx) in newIndividualGen:
			newIndividual = bounder(newIndividual)
			newValue = self.safeEvaluate(self.problemID, newIndividual)
			if newValue < value:
				self.population[idx] = newIndividual
				self.values[idx] = newValue
			if self.safeGetEvaluations(self.problemID) >= self.budget:
				return

		newVar = np.var(self.population, axis = 0)
		c = self.var / newVar
		self.var = newVar

		for i, ci in enumerate(c):
			ti = (1. - self.CP)**2 / self.popNum + (self.popNum - 1.) / self.popNum
			if ci < ti:
				self.F[i] = self.flb
			else:
				self.F[i] = np.sqrt((ci - ti) / (2*self.CP))
			self.F[i] = max(min(self.F[i], self.fub), self.flb)

	def initParams(self):
		self.F = np.ones(self.dimNum) * np.sqrt(1. / self.popNum)
		self.CP = 0.5
		self.flb, self.fub = np.sqrt(1. / self.popNum), 2.
		self.var = np.var(self.population, axis = 0)

class OBDE(EA):

	def __init__(self, popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations):
		super(OBDE, self).__init__(popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations)
		self.OBPI()

	def optimizeOneIter(self):
		parents, pValues, pIdxes = uniformSelection(self.population, self.values, self.popNum)

		for individual, value, idx in zip(parents, pValues, pIdxes):
			neighborIdxes = range(self.popNum)
			del neighborIdxes[idx]
			variant = rand_1_variation(self.population, neighborIdxes, self.F)
			newIndividual = DEBinomialCrossover(individual, variant, self.CP)
			newIndividual = bounder(newIndividual)
			newValue = self.safeEvaluate(self.problemID, individual)

			if newValue < value:
				self.population[idx] = newIndividual
				self.values[idx] = newValue

			if self.safeGetEvaluations(self.problemID) >= self.budget:
				return

		if np.random.rand() < self.JR:
			self.OBGJ()

	def OBPI(self):
		''' Opposition-Based Population Initialization '''
		oppositionPopulation = np.zeros(shape = (self.popNum, self.dimNum))
		oppositionValues = np.zeros(self.popNum)
		for i, individual in enumerate(self.population):
			oppositionPopulation[i] = self.getOpposition(individual, 0, 1)
			oppositionValues[i] = self.safeEvaluate(self.problemID, individual)
			if self.safeGetEvaluations(self.problemID) >= self.budget:
				return
		self.population, self.values = plusReplacement(
			self.population,
			self.values,
			oppositionPopulation,
			oppositionValues
		)

	def OBGJ(self):
		''' Opposition-Based Generation Jumping '''
		lbs, ubs = np.min(self.population, axis = 0), np.max(self.population, axis = 0)
		oppositionPopulation = np.zeros(shape = (self.popNum, self.dimNum))
		oppositionValues = np.zeros(self.popNum)
		for i, individual in enumerate(self.population):
			oppositionPopulation[i] = self.getOpposition(individual, lbs, ubs)
			oppositionValues[i] = self.safeEvaluate(self.problemID, individual)
			if self.safeGetEvaluations(self.problemID) >= self.budget:
				return
		self.population, self.values = plusReplacement(
			self.population,
			self.values,
			oppositionPopulation,
			oppositionValues
		)


	def initParams(self):
		self.CP = 0.9
		self.JR = 0.3
		self.F = 0.5

	@staticmethod
	def getOpposition(individual, lbs, ubs):
		return lbs + ubs - individual

		

class CMAES(EA):

	def __init__(self, popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations):
		self.popNum = popNum
		self.dimNum = dimNum
		self.problemID = problemID
		self.budget = budget

		self.safeEvaluate = safeEvaluate
		self.safeGetEvaluations = safeGetEvaluations

		self.initParams()

	def optimizeOneIter(self):
		arz 			= np.random.randn(self.dimNum, self.lambd)
		arx 			= np.zeros(shape = (self.dimNum, self.lambd))
		for k in range(self.lambd):
			arx[:, k] 	= self.xmean.flatten() + self.sigma * np.dot(self.B, (self.D.flatten() * arz[:, k]))

		row, col = arx.shape
		for i in range(row):
			for j in range(col):
				if arx[i, j] > 1:
					arx[i, j] = fmod(arx[i, j], 1)
				elif arx[i, j] < 0:
					arx[i, j] = fmod(abs(arx[i, j]), 1)

		self.population = arx.T
		self.values = np.zeros(self.lambd)
		for i, individual in enumerate(self.population):
			self.values[i] = self.safeEvaluate(self.problemID, np.array(list(individual)))
			if self.safeGetEvaluations(self.problemID) >= self.budget:
				return

		kBestIndex 		= self.values.argsort()[:self.mu]
		xold 			= copy(self.xmean)
		self.xmean 		= np.dot(arx[:, kBestIndex], self.weights).reshape(self.dimNum, 1)
		self.zmean 		= np.dot(arz[:, kBestIndex], self.weights).reshape(self.dimNum, 1)
		self.ps 		= (1. - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.dot(self.invsqrtC, (self.xmean - xold) / self.sigma)
		hsig 		= np.linalg.norm(self.ps) / np.sqrt(1. - (1. - self.cs)**(2. * self.safeGetEvaluations(self.problemID) / self.lambd)) / self.chiN < 1.4 + 2. / (self.dimNum + 1.)
		self.pc 		= (1. - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2. - self.cc) * self.mueff) * ((self.xmean - xold) / self.sigma)
		artmp 			= (1./ self.sigma) * (arx[:, kBestIndex] - xold)
		self.C 			= (1 - self.c1 - self.cmu) * self.C +\
							 self.c1 * (np.dot(self.pc, self.pc.T) + (1. - hsig) * self.cc * (2. - self.cc) * self.C) +\
							 self.cmu * np.dot(np.dot(artmp, np.diag(self.weights.flatten())), artmp.T)
		self.sigma 		= self.sigma * np.exp((self.cs / self.damps) * (1. * np.linalg.norm(self.ps) / self.chiN - 1.)) 

		if self.safeGetEvaluations(self.problemID) - self.eigeneval > self.lambd / (self.c1 + self.cmu) / self.dimNum / 10:
			self.eigeneval 	= self.safeGetEvaluations(self.problemID)
			self.C 			= np.triu(self.C) + np.triu(self.C, 1).T
			self.D, self.B 	= np.linalg.eigh(self.C) # D: eigenvalues, shape = (1, n); B: eigenvectors, shape = (d, n)
			self.D 			= np.sqrt(self.D).reshape(self.dimNum, 1)
			self.invsqrtC 	= np.dot(np.dot(self.B, np.diag(self.D.flatten()**-1)), self.B.T)


	def initParams(self):
		# user defined input parameters
		self.xmean 		= np.random.rand(self.dimNum, 1)
		self.sigma 		= 0.3
		# strategy parameter setting: Selection
		self.lambd 		= 4 + int(np.floor(3 * np.log(self.dimNum))) # population size, offspring number
		self.mu 		= self.lambd / 2 # lambda = 12, mu = 3, weights = np.ones(mu, 1) would be (3_I, 12) -ES
		self.weights 	= np.log(self.mu + 0.5) - np.log(np.arange(self.mu) + 1)
		self.mu 		= int(self.mu)
		self.weights 	= (self.weights / np.sum(self.weights)).reshape(self.mu, 1)
		self.mueff 		= (np.sum(self.weights))**2 / np.sum(self.weights**2) # variance-effective size of mu

		# strategy parameter setting: Adaptation
		self.cc 		= (4 + 1. * self.mueff / self.dimNum) / (self.dimNum + 4 + 2. * self.mueff / self.dimNum) # time constant for cumulation for C
		self.cs 		= (self.mueff + 2) / (self.dimNum + self.mueff + 5) # t-constant for cumulation for sigma control
		self.c1 		= 2. / ((self.dimNum + 1.3)**2 + self.mueff) # learning rate for rank-one update of C
		#self.cmu 		= 2 * (self.mueff - 2 + 1./ self.mueff) / ((self.dimNum + 2)**2 + 2. * self.mueff / 2) # and for rank-mu update
		self.cmu 		= min(1 - self.c1, 2 * (self.mueff - 2. + 1./self.mueff) / ((self.dimNum + 2)**2 + self.mueff))
		self.damps 		= 1 + 2 * max(0, np.sqrt(1. * (self.mueff - 1)/(self.dimNum + 1)) - 1) + self.cs # damping for sigma

		# initialize dynamic (internal) strategy parameters and constants
		self.pc 		= np.zeros(shape = (self.dimNum, 1)) # evolution paths for C
		self.ps 		= np.zeros(shape = (self.dimNum, 1)) # evolution paths for sigma
		self.B  		= np.eye(self.dimNum) # B defines the coordinate system
		self.D 			= np.ones(shape = (self.dimNum, 1))
		self.C 			= np.dot(np.dot(self.B, np.diag(self.D.flatten()**2)), self.B.T)
		self.invsqrtC 	= np.dot(np.dot(self.B, np.diag(self.D.flatten()**-1)), self.B.T)
		self.eigeneval 	= 0 # B and D updated at counteval == 0
		self.chiN 		= self.dimNum**0.5 * (1 - 1./ (4 * self.dimNum) + 1./(21 * (self.dimNum**2))) #expectation of ||N(0, I)|| == norm(randn(N, 1))


class SOUPDE(EA):

	def __init__(self, popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations):
		super(SOUPDE, self).__init__(popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations)

	def optimizeOneIter(self):
		for i in range(self.numSBP):
			start, end = i * self.numPSP, (i + 1) * self.numPSP
			# population, values, parents, pValues, pIdxes, f, cp
			newIndividualGen = rand_1_exp(
				self.population[start: end],
				self.values[start: end],
				self.population[start: end],
				self.values[start: end],
				range(self.numPSP),
				float(self.F[i]),
				self.CP
			)
			for (newIndividual, value, idx) in newIndividualGen:
				newIndividual = bounder(newIndividual)
				newValue = self.safeEvaluate(self.problemID, newIndividual)
				if newValue < value:
					self.population[start + idx] = newIndividual
					self.values[start + idx] = newValue
				if self.safeGetEvaluations(self.problemID) >= self.budget:
					return
		if np.random.rand() < self.PS:
			self.shuffle()
		if np.random.rand() < self.PU:
			self.F = self.genScaleFactor()

	def initParams(self):
		self.PS = 0.5 # probability of the shuffling operation
		self.PU = 0.1 # probability of the update operation
		self.numSBP =3 # number of sub-populations
		self.numPSP = self.popNum / self.numSBP # number of members per sub-population
		self.F = self.genScaleFactor()
		self.CP = 0.9

	def genScaleFactor(self):
		return 0.1 + np.random.rand(self.numSBP)

	def shuffle(self):
		permutation = np.random.permutation(self.popNum)
		self.population = self.population[permutation]
		self.values = self.values[permutation]

class SPSO2011(EA):

	def __init__(self, popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations):
		super(SPSO2011, self).__init__(popNum, dimNum, problemID, budget, safeEvaluate, safeGetEvaluations)
		self.initOthers()
		self.initParams()

	def optimizeOneIter(self):
		if self.params_init_links:
			self.initLinks()

		improve = False
		for s, (fea, vlt, p_fea, val, lnk) in enumerate(zip(self.population, self.velocity, self.pBest, self.values, self.links)):
			# find the first informant
			s1 = 0
			while lnk[s1] == 0:
				s1 += 1
			if s1 >= self.popNum:
				s1 = s
			# find best informant
			g = s1
			for m in range(s1, self.popNum):
				if (lnk[m] == 1 and self.values[m] < self.values[g]):
					g = m
			# compute the new velocity and move
			## Exploration tendency
			vlt[:] = self.params_w * vlt # this will change the velocity in self.velocity too!!
			## Exploitation tendency p-x
			px_vlt = p_fea - fea
			## if the particle is not its own local best, prepare g-x
			if g != s:
				gx_vlt = self.pBest[g] - fea
			else:
				s1 = np.random.randint(self.popNum)
				while s1 == s:
					s1 = np.random.randint(self.popNum)
			## Gravity center
			w1, w2, w3 = 1., 1., 1.
			if g == s:
				w3 = 0.
			zz = w1 + w2 + w3
			w1, w2, w3 = w1/zz, w2/zz, w3/zz
			if g != s:
				gr = w1 * fea + w2 * (fea + self.params_c * px_vlt) + w3 * (fea + self.params_c * gx_vlt)
			else:
				gr = w1 * fea + w2 * (fea + self.params_c * px_vlt)
			V1 = gr - fea 
			### random point around
			rad = np.sqrt(np.sum((gr - fea) ** 2))
			V2 	= self.aleaSphere(self.dimNum, rad)
			# New 'velocity'
			vlt[:] = vlt + V1 + V2
			# New position
			fea[:] = fea + vlt
			# confinement
			ud_flow = fea < 0.
			up_flow = fea > 1.
			fea[:]  = ud_flow * 0. + \
					  up_flow * 1. + \
					  (1 - ud_flow - up_flow) * fea
			vlt[:] 	= (ud_flow + up_flow) * (-0.5) * vlt +\
					  (1 - ud_flow - up_flow) * vlt
			new_val = self.safeEvaluate(self.problemID, fea)
			if self.safeGetEvaluations(self.problemID) >= self.budget:
				return
			if new_val < val:
				p_fea[:] = fea[:]
				self.values[s] = new_val
				if new_val < min(self.values):
					improve = True
		if improve:
			self.params_init_links = False
		else:
			self.params_init_links = True
		return self.popNum

	@staticmethod
	def aleaSphere(D, radius):
		x = np.random.randn(D)
		v = radius * np.random.rand() * x / np.sqrt(np.sum(x**2))
		return v

	def initOthers(self):
		self.velocity = (0. - self.population) + \
						(1. - self.population) * np.random.rand(self.popNum, self.dimNum)
		self.pBest   = copy(self.population)

	def initParams(self):
		self.params_K = 3 # parameter to compute the probability p for a particle to be an external informant
		self.params_w = 1. / (2. * np.log(2.))
		self.params_c = 0.5 + np.log(2.)
		self.params_S = self.popNum
		self.params_p = 1. - np.power(1 - 1. / self.params_S, self.params_K)

		self.params_init_links = True

	def initLinks(self):
		eyeMask 	= np.eye(self.popNum)
		tmpLinks	= np.random.rand(self.popNum, self.popNum) < self.params_p
		self.links 	= eyeMask + (1. - eyeMask) * tmpLinks


def bounder(ind):
	# reinitialize
	newInd = copy(ind)
	for i, d in enumerate(ind):
		if (d > 1.0 or d < 0.0):
			newInd[i] = np.random.rand()
	return newInd