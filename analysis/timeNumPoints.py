from timeSummary import gatherData, printSummary

if __name__ == "__main__":
	headings, results = gatherData("#numPoints", "bin/timeNumPoints", 30)
	printSummary("numPoints", headings, results)

