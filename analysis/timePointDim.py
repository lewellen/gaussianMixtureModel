from timeSummary import gatherData, printSummary

if __name__ == "__main__":
	headings, results = gatherData("pointDim", "bin/timePointDim", 15)
	printSummary("pointDim", headings, results)

