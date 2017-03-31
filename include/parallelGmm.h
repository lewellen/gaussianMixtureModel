#ifndef PARALLELGMM_H
#define PARALLELGMM_H

#include <stdlib.h>

#include "gmm.h"

struct GMM* parallelFit(
	const double* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents
);

#endif 
