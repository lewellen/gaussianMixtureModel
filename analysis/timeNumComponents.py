from timeSummary import gatherData, printSummary

if __name__ == "__main__":
	headings, results = gatherData("numComponents", "bin/timeNumComponents", 30)
	printSummary("numComponents", headings, results)

