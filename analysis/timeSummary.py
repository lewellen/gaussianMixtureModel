import subprocess
import StringIO
import csv

import numpy
from scipy import stats

def gatherData(primaryKey, execPath, desiredRuns):
	successfulRuns = 0

	num = primaryKey
	seq = "seqElapsedSec"
	par = "parallelElapsedSec"
	cud = "cudaElapsedSec"

	headings = [seq, par, cud]

	results = { }

	while successfulRuns < desiredRuns:
		proc = subprocess.Popen([execPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
		out, err = proc.communicate()

		if proc.returncode != 0:
			continue

		successfulRuns += 1

		f = StringIO.StringIO(out)
		reader = csv.DictReader(f, delimiter=' ')
		for row in reader:
			key = float(row[num])
			if not key in results:
				results.update( { key : { seq : [], par : [], cud : [] } } )

			for heading in headings:
				results[key][heading].append( float(row[heading]) )

	return headings, results

def printSummary(primaryKey, headings, results):
	print("# %s " % primaryKey),
	for heading in headings:
		print("%s CILB CIUB " % heading),
	print("")

	sortedKeys = sorted(results.keys())
	for numPoints in sortedKeys:
		print numPoints,
		for strategy in results[numPoints]:
			xs = numpy.array( results[numPoints][strategy] )
			xs = sorted(xs)
			xs = xs[int(0.025 * len(xs)):int(0.975 * len(xs))]

			sampleMean = numpy.mean(xs)
			sampleStd = numpy.std(xs)
			confInt = stats.norm.interval(0.95, sampleMean, sampleStd)
			print("%f %f %f" % (sampleMean, max(0, confInt[0]), max(0, confInt[1]))),
		print("")
