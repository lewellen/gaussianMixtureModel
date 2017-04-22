import subprocess
import StringIO
import csv

import math
import numpy
from scipy import stats

def gatherData(primaryKey, columns, execPath, desiredRuns):
	successfulRuns = 0

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
			key = float(row[primaryKey])
			if not key in results:
				results.update( { key : { column : [] for column in columns } } )

			for column in columns:
				results[key][column].append( float(row[column]) )

	return results

def printSummary(primaryKey, columns, displayColumns, results):
	print("%s " % primaryKey),
	for column in displayColumns:
		print("%s CILB CIUB " % column),
	print("")

	sortedKeys = sorted(results.keys())
	for row in sortedKeys:
		print row,
		for column in columns:
			xs = numpy.array( results[row][column] )
			xs = sorted(xs)

			n = len(xs)
			lb = int(math.floor(0.025 * n))
			if lb < 0:
				lb = 0
			ub = int(math.ceil(0.0975 * n))
			if ub > n - 1:
				ub = n -1

			xs = xs[lb:ub]

			sampleMean = numpy.mean(xs)
			sampleStd = numpy.std(xs)
			if n <= 30:
				confInt = stats.t.interval(0.95, n, sampleMean, sampleStd)
			else:
				confInt = stats.norm.interval(0.95, sampleMean, sampleStd)

			print("%f %f %f" % (sampleMean, max(0, confInt[0]), max(0, confInt[1]))),
		print("")
