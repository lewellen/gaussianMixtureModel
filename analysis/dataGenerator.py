from sklearn.datasets.samples_generator import make_blobs

if __name__ == '__main__':
	X, Y = make_blobs(n_samples = 10000, n_features=5, centers=16)	

	print("# x0 x1")
	for x in X:
		print("%f %f" % (x[0], x[1]))

	pass
