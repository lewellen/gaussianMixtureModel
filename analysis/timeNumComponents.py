from timeSummary import gatherData, printSummary

if __name__ == "__main__":
	execPath = "bin/timeNumComponents"
	numSamples = 15

	primaryKey = "numComponents"
	columns = [ "seqElapsedSec", "parallelElapsedSec", "cudaElapsedSec" ]
	displayColumns = [ "Sequential", "Parallel", "GPU" ]

	results = gatherData(primaryKey, columns, execPath, numSamples)
	printSummary(primaryKey, columns, displayColumns, results)

