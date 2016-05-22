from ctypes import *
from numpy.ctypeslib import ndpointer
from numpy import *
import sys
import platform
import time
from slg import SLG
from portfolio import *
from helpers import dim2PopNum

# get library name
dllname = ""
if platform.system() == "Windows":
	dllname = "./bbcomp.dll"
elif platform.system() == "Linux":
	dllname = "./libbbcomp.so"
elif platform.system() == "Darwin":
	dllname = "./libbbcomp.dylib"
else:
	sys.exit("unknown platform")

# initialize dynamic library
bbcomp = CDLL(dllname)
bbcomp.configure.restype = c_int
bbcomp.login.restype = c_int
bbcomp.numberOfTracks.restype = c_int
bbcomp.trackName.restype = c_char_p
bbcomp.setTrack.restype = c_int
bbcomp.numberOfProblems.restype = c_int
bbcomp.setProblem.restype = c_int
bbcomp.dimension.restype = c_int
bbcomp.numberOfObjectives.restype = c_int
bbcomp.budget.restype = c_int
bbcomp.evaluations.restype = c_int
bbcomp.evaluate.restype = c_int
bbcomp.evaluate.argtypes = [ndpointer(c_double, flags="C_CONTIGUOUS"), ndpointer(c_double, flags="C_CONTIGUOUS")]
bbcomp.history.restype = c_int
bbcomp.history.argtypes = [c_int, ndpointer(c_double, flags="C_CONTIGUOUS"), ndpointer(c_double, flags="C_CONTIGUOUS")]
bbcomp.errorMessage.restype = c_char_p

print "----------------------------------------------"
print "black box example competition client in Python"
print "----------------------------------------------"
print

# configuration
LOGFILEPATH = 'logs/'
USERNAME = 'hezheng'
PASSWORD = 'hezheng562'
TRACKNAME = 'trial'
LOGIN_DELAY_SECONDS = 10
LOCK_DELAY_SECONDS = 60

# network failure resilient functions
def safeLogin():
	while True:
		result = bbcomp.login(USERNAME, PASSWORD)
		if result != 0:
			return
		msg = bbcomp.errorMessage()
		print "WARNING: login failed: ", msg
		time.sleep(LOGIN_DELAY_SECONDS)
		if msg == "already logged in":
			return

def safeSetTrack():
	while True:
		result = bbcomp.setTrack(TRACKNAME)
		if result != 0:
			return
		print "WARNING: setTrack failed: ", bbcomp.errorMessage()
		safeLogin()

def safeGetNumberOfProblems():
	while True:
		result = bbcomp.numberOfProblems()
		if result != 0:
			return result
		print "WARNING: numberOfProblems failed: ", bbcomp.errorMessage()
		safeSetTrack()

def safeSetProblem(problemID):
	while True:
		result = bbcomp.setProblem(problemID)
		if result != 0:
			return
		msg = bbcomp.errorMessage()
		print "WARNING: setProblem failed: ", msg
		if len(msg) >= 22 and msg[0:22] == "failed to acquire lock":
			time.sleep(LOCK_DELAY_SECONDS)
		else:
			safeSetTrack()

def safeEvaluate(problemID, point):
	while True:
		value = zeros(1)
		result = bbcomp.evaluate(point, value)
		if result != 0: return value[0]
		print "WARNING: evaluate failed: ", bbcomp.errorMessage()
		safeSetProblem(problemID)

def safeGetDimension(problemID):
	while True:
		result = bbcomp.dimension()
		if result != 0: return result
		print "WARNING: dimension failed: ", bbcomp.errorMessage()
		safeSetProblem(problemID)

def safeGetBudget(problemID):
	while True:
		result = bbcomp.budget()
		if result != 0: return result
		print "WARNING: budget failed: ", bbcomp.errorMessage()
		safeSetProblem(problemID)

def safeGetEvaluations(problemID):
	while True:
		result = bbcomp.evaluations()
		if result >= 0: return result
		print "WARNING: evaluations failed: " + bbcomp.errorMessage()
		safeSetProblem(problemID)

def solveProblem(problemID):
	# set the problem
	safeSetProblem(problemID)

	# obtain problem properties
	bud = safeGetBudget(problemID)
	dim = safeGetDimension(problemID)
	evals = safeGetEvaluations(problemID)

	# output status
	if evals == bud:
		print "problem ", problemID, ": already solved"
		return
	elif evals == 0:
		print "problem ", problemID, ": starting from scratch"
	else:
		print "problem ", problemID, ": starting from evaluation ", evals

	# SLG algorithm


	# optimizor = OBDE(40, dim, problemID, bud, safeEvaluate, safeGetEvaluations)
	# optimizor = SOUPDE(40, dim, problemID, bud, safeEvaluate, safeGetEvaluations)	
	# optimizor = jDE(40, dim, problemID, bud, safeEvaluate, safeGetEvaluations)	
	# optimizor = DZAdaptiveDE(40, dim, problemID, bud, safeEvaluate, safeGetEvaluations)	
	optimizor = SPSO2011(40, dim, problemID, bud, safeEvaluate, safeGetEvaluations)	
	optimizor.optimize()

# setup
result = bbcomp.configure(1, 'logs/')
if result == 0:
	sys.exit("configure() failed: " + bbcomp.errorMessage())

safeLogin()
safeSetTrack()
n = safeGetNumberOfProblems()

for i in range(n):
	solveProblem(i)