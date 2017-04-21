import matplotlib.pyplot as plt
import numpy
import sys
import re

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

def sideBySideSolverComparison(dataFilePath, numComponents, solvers, solverNames, models, outputPng):
	cMap = makeColorMap(numComponents)

	data = numpy.loadtxt(dataFilePath)
	X = data[:, 0]
	Y = data[:, 1]

	fig, axes = plt.subplots(1, len(solvers), sharey=True, sharex=True, figsize=(10, 4))
	for (index, model) in zip(xrange(0, len(models)), models):
		axes[index].set_title("%s (%.6f sec)" % (solverNames[index], model.elapsedSec))
		axes[index].set_xlabel('X')
		axes[index].set_ylabel('Y')
		axes[index].scatter(X, Y, s=0.5)
		plotGmm(axes[index], fig, model.comps, cMap)

	plt.xlim([min(X) - 0.01 * abs(min(X)), max(X) + 0.01 * abs(max(X))])
	plt.ylim([min(Y) - 0.01 * abs(min(Y)), max(Y) + 0.01 * abs(max(Y))])

	plt.savefig(outputPng, format='png', dpi=720)

if __name__ == "__main__":

	if len(sys.argv) in [3, 4]:
		dataFilePath = sys.argv[1]
		if len(sys.argv) == 3:
			m = re.findall('k(\d+)', sys.argv[1])
			if len(m) == 0:
				print("%s <dataFilePath> <numComponents> <output.png>" % (sys.argv[0]))
				exit(1)
			

			numComponents = int(m[0])
			outputPng = sys.argv[2]
		else:
			numComponents = int(sys.argv[2])
			outputPng = sys.argv[3]
	else:
		print("%s <dataFilePath> <numComponents> <output.png>" % (sys.argv[0]))
		exit(1)


	solvers, solverNames, models = gatherData(dataFilePath, numComponents)
	sideBySideSolverComparison(dataFilePath, numComponents, solvers, solverNames, models, outputPng)
