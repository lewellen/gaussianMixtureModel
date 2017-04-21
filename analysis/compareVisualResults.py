import matplotlib.pyplot as plt
import numpy
import sys

from cmdLineWrappers import GMMSolver, GMM, Component
from plotUtils import makeColorMap, plotGmm

def gatherData(dataFilePath, numComponents):
	solvers = [ GMMSolver.Sequential, GMMSolver.Parallel, GMMSolver.CUDA ]
	solverNames = [ "Sequential", "Parallel", "CUDA" ]
	models = []

	for solver in solvers:
		model = GMM(solver)
		model.fit(dataFilePath, numComponents)
		models.append(model)

	return solvers, solverNames, models

def sideBySideSolverComparison(dataFilePath, numComponents, solvers, solverNames, models):
	cMap = makeColorMap(numComponents)

	data = numpy.loadtxt(dataFilePath)
	X = data[:, 0]
	Y = data[:, 1]

	fig, axes = plt.subplots(1, len(solvers), sharey=True, sharex=True)
	for (index, model) in zip(xrange(0, len(models)), models):
		axes[index].set_title("%s (%.6f sec)" % (solverNames[index], model.elapsedSec))
		axes[index].set_xlabel('X')
		axes[index].set_ylabel('Y')
		axes[index].scatter(X, Y, s=1)
		plotGmm(axes[index], fig, model.comps, cMap)

	plt.xlim([min(X) - 0.01 * abs(min(X)), max(X) + 0.01 * abs(max(X))])
	plt.ylim([min(Y) - 0.01 * abs(min(Y)), max(Y) + 0.01 * abs(max(Y))])

	plt.show()

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("%s <dataFilePath> <numComponents>" % (sys.argv[0]))
		exit(1)

	dataFilePath = sys.argv[1]
	numComponents = int(sys.argv[2])

	solvers, solverNames, models = gatherData(dataFilePath, numComponents)
	sideBySideSolverComparison(dataFilePath, numComponents, solvers, solverNames, models)

