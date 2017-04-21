#ifndef KMEANS_H
#define KMEANS_H

void kmeans(
	const double* X, const size_t numPoints, const size_t pointDim, 
	double* M, const size_t numComponents
); 

#endif
