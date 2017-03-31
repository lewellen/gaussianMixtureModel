#ifndef SEQGMM_H
#define SEQGMM_H

#include <stdlib.h>

#include "gmm.h"

struct GMM* fit(
	const double* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents
);

#endif 
