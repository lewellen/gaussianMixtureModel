from timeSummary import gatherData, printSummary

if __name__ == "__main__":
	execPath = "bin/timeNumPoints"
	numSamples = 5

	primaryKey = "#numPoints"
	columns = [ "seqElapsedSec", "parallelElapsedSec", "cudaElapsedSec" ]
	displayColumns = [ "Sequential", "Parallel", "GPU" ]

	results = gatherData(primaryKey, columns, execPath, numSamples)
	printSummary(primaryKey.replace("#", ""), columns, displayColumns, results)

