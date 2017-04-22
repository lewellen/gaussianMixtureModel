import numpy
import matplotlib.pyplot as plt

if __name__ == "__main__":
	fileNames = [ "obj/timeNumPoints-summary.dat", "obj/timePointDim-summary.dat", "obj/timeNumComponents-summary.dat" ]
	columns = numpy.array( [ 0, 1, 2] )
	columnNames = [ "$N$", "$d$", "$K$" ]

	parallelValues = []
	cudaValues = []

	for fileName in fileNames:
		data = numpy.loadtxt(fileName, skiprows=1)

		seq = data[:, 1]
		par = data[:, 4]
		cuda = data[:, 7]

		cMax = numpy.max(numpy.divide(seq, cuda))
		pMax = numpy.max(numpy.divide(seq, par))

		parallelValues.append(pMax)
		cudaValues.append(cMax)

	plt.xticks(columns, columnNames)
	plt.ylabel('Maximum Speedup, $x$ (alternative = x baseline)')
	plt.bar(columns - 0.2,  parallelValues, width=0.4, color='red')
	plt.bar(columns + 0.2, cudaValues, width=0.4, color='green')
	plt.legend(('Parallel', 'CUDA'))
	plt.savefig('obj/speedup.eps', format='eps', dpi=720)
