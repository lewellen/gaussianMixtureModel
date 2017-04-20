#ifndef PARALLELGMM_H
#define PARALLELGMM_H

#include <stdlib.h>

#include "barrier.h"
#include "gmm.h"

// These are resources accessed by all threads.
struct SharedThreadStartArgs {
	const double* X;
	size_t numPoints;
	size_t pointDim;

	struct GMM* gmm;

	double tolerance;
	size_t maxIterations;
	double prevLogL;
	double currentLogL;

	int shouldContinue;

	double* logpi;
	double* loggamma;
	double* logGamma;
	double logGammaSum;

	double* logLK;
	size_t numProcesses;

	struct Barrier* barrier;
};

// These are resources assigned to a single thread.
struct ThreadStartArgs {
	size_t id;
	struct SharedThreadStartArgs* shared;		

	// These limits apply only to computing gamma
	size_t pointStart;
	size_t pointEnd; // i = pointStart; i < pointEnd; ++i

	// These limits apply for all components
	size_t componentStart;
	size_t componentEnd; // i = componentStart; i < componentEnd; ++i
};

// Barrier critical section callbacks
void checkStopCriteria(
	void* untypedArgs
);

void computeGammaSum(
	void* untypedArgs
);

// Thread start
void* parallelFitStart(
	void* untypedArgs
);

struct GMM* parallelFit(
	const double* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents,
	const size_t maxIterations
);

#endif 
