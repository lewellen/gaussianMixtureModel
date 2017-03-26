from sklearn.datasets.samples_generator import make_blobs

if __name__ == '__main__':
	X, Y = make_blobs(n_samples = 500, n_features=2, centers=4)	

	print("# x0 x1")
	for x in X:
		print("%f %f" % (x[0], x[1]))

	pass
