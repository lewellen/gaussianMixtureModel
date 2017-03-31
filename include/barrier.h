#ifndef BARRIER_H
#define BARRIER_H

#include <stdlib.h>

#include <sys/types.h>
#include <pthread.h>

struct Barrier {
	pthread_cond_t cv;
	pthread_mutex_t mx;
	size_t totalProcessors;
	size_t waitingProcessors;
};

void initBarrier(
	struct Barrier* barrier, 
	size_t headCount
); 

void destroyBarrier(
	struct Barrier* barrier
); 

void arriveAt(
	struct Barrier* barrier, 
	void* arg, 
	void (*callback)(void*) 
); 

#endif
