#include <assert.h>
#include <stdlib.h>

#include <sys/types.h>
#include <pthread.h>

#include "barrier.h"

void initBarrier(struct Barrier* barrier, size_t headCount) {
	assert(barrier != NULL);
	assert(headCount > 0);

	pthread_cond_init(&barrier->cv, NULL);
	pthread_mutex_init(&barrier->mx, NULL);
	barrier->waitingProcessors = 0;
	barrier->totalProcessors = headCount;
}

void destroyBarrier(struct Barrier* barrier) {
	assert(barrier != NULL);
	pthread_mutex_destroy(&barrier->mx);
	pthread_cond_destroy(&barrier->cv);
}

void arriveAt(struct Barrier* barrier, void* arg, void (*callback)(void*) ) {
	assert(barrier != NULL);

	pthread_mutex_lock(&barrier->mx);

	if( ++ barrier->waitingProcessors == barrier->totalProcessors ) {
		barrier->waitingProcessors = 0;
		if(callback != NULL) {
			// Call back for whoever enters the barrier
			callback(arg);
		}
		pthread_cond_broadcast(&barrier->cv);
	} else {
		pthread_cond_wait(&barrier->cv, &barrier->mx);
		// lock on mx reacquired
	}

	pthread_mutex_unlock(&barrier->mx);
}
