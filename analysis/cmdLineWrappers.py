import json
import subprocess

class Component:
	def __init__(self):
		self.pi = 0;
		self.mu = []
		self.sigma = []

class GMMSolver:
	Sequential = 1
	Parallel = 2
	CUDA = 3

class GMM:
	def __init__(self, gmmSolver):
		execs = {
			GMMSolver.Sequential : "bin/sequentialGmm",
			GMMSolver.Parallel : "bin/parallelGmm",
			GMMSolver.CUDA : "bin/cudaGmm"
		}

		self.elapsedSec = None

		if gmmSolver in execs:
			self.solver = execs[gmmSolver]
		else:
			self.solver = execs[GMMSolver.Sequentail]

		self.comps = []

	def fit(self, dataPath, numComponents):
		args = [ self.solver, dataPath, str(numComponents) ]

		proc = subprocess.Popen(args, 
			stdout=subprocess.PIPE, 
			stderr=subprocess.PIPE, 
			shell=False
			)

		out, err = proc.communicate()
		if proc.returncode != 0:
			print err
			return False

		result = json.loads(out)
		self.comps = result["model"]["mixtures"]
		self.elapsedSec = float(result["elapsedSec"])

		return True
