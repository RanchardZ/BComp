
from ctypes import *
from numpy.ctypeslib import ndpointer
from numpy import *
import sys
import platform

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

# change the track name to trialMO for multi-objective optimization
track = "trial"
# track = "trialMO"

# set configuration options (this is optional)
result = bbcomp.configure(1, "logs/")
if result == 0:
	sys.exit("configure() failed: " + bbcomp.errorMessage())

# login with demo account - this should grant access to the "trial" and "trialMO" tracks (for testing and debugging)
result = bbcomp.login("demoaccount", "demopassword")
if result == 0:
	sys.exit("login() failed: " + bbcomp.errorMessage())

print "login successful"

# request the tracks available to this user (this is optional)
numTracks = bbcomp.numberOfTracks()
if numTracks == 0:
	sys.exit("numberOfTracks() failed: " + bbcomp.errorMessage())

print numTracks, " track(s):"
for i in range(numTracks):
	trackname = bbcomp.trackName(i)
	if bool(trackname) == False:
		sys.exit("trackName() failed: " + bbcomp.errorMessage())

	print "  ", i, ": ", trackname

# set the track
result = bbcomp.setTrack(track)
if result == 0:
	sys.exit("setTrack() failed: " + bbcomp.errorMessage())

print "track set to ", track

# obtain number of problems in the track
numProblems = bbcomp.numberOfProblems()
if numProblems == 0:
	sys.exit("numberOfProblems() failed: " << bbcomp.errorMessage())

print "The track consists of ", numProblems, " problems."

# For demonstration purposes we optimize only the first problem in the track.
problemID = 0
result = bbcomp.setProblem(problemID)
if result == 0:
	sys.exit("setProblem() failed: " + bbcomp.errorMessage())

print "Problem ID set to ", problemID

# obtain problem properties
dim = bbcomp.dimension()
if dim == 0:
	sys.exit("dimension() failed: " + bbcomp.errorMessage())

obj = bbcomp.numberOfObjectives()
if dim == 0:
	sys.exit("numberOfObjectives() failed: " + bbcomp.errorMessage())

bud = bbcomp.budget()
if bud == 0:
	sys.exit("budget() failed: " + bbcomp.errorMessage())

evals = bbcomp.evaluations()
if evals < 0:
	sys.exit("evaluations() failed: " + bbcomp.errorMessage())

print "problem dimension: ", dim
print "number of objectives: ", obj
print "problem budget: ", bud
print "number of already used up evaluations: ", evals

# allocate memory for a search point
point = zeros(dim)
value = zeros(obj)

# run the optimization loop
for e in range(bud):
	if e < evals:
		# If evals > 0 then we have already optimized this problem to some point.
		# Maybe the optimizer crashed or was interrupted.
		#
		# This code demonstrates a primitive recovery approach, namely to replay
		# the history as if it were the actual black box queries. In this example
		# this affects only "bestValue" since random search does not have any
		# state variables.
		# As soon as e >= evals we switch over to black box queries.
		result = bbcomp.history(e, point, value)
		if result == 0:
			sys.exit("history() failed: " + bbcomp.errorMessage())
	else:
		# define a search point, here uniformly at random
		point = random.rand(dim);

		# query the black box
		result = bbcomp.evaluate(point, value)
		if result == 0:
			sys.exit("evaluate() failed: " + bbcomp.errorMessage())

	# In any real algorithm "point" and "value" would update the internals state.
	# Here we just output the values.
	print '[{0}] f({1}) = {2}'.format(e, point, value)
	# print "[", e, "] f(".join(point).join(") = ").join(value)

# check that we are indeed done
evals = bbcomp.evaluations()
if evals == bud:
	print "optimization finished."
else:
	print "something went wrong: number of evaluations does not coincide with budget :("
