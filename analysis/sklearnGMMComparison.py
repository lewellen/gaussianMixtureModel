
import numpy
import sys
from sklearn.mixture import GaussianMixture

if __name__ == '__main__':
	argc = len(sys.argv)
	if argc == 1:
		print "%s <.dat>" % sys.argv[0]
		exit(1)

	datFilePath = sys.argv[1]
	X = numpy.loadtxt(datFilePath)

	gmm = GaussianMixture(n_components=2)
	gmm.fit(X)

	print "weights: ",
	print gmm.weights_

	print "means: ",
	print gmm.means_

	print "covariances: ",
	print gmm.covariances_
