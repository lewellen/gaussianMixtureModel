#ifndef CUDAGMM_H
#define CUDAGMM_H

#include <stdlib.h>

#include "gmm.h"

struct GMM* cudaFit(
	const double* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents,
	const size_t maxIterations
);

#endif 
