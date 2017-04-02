#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include "util.h"

double calcElapsedSec(struct timeval* start, struct timeval* stop) {
	assert(start != NULL);
	assert(stop != NULL);

	double sec = stop->tv_sec - start->tv_sec;
	double usec = stop->tv_usec - start->tv_usec;
	if(stop->tv_sec > start->tv_sec) {
		if(start->tv_usec > stop->tv_usec) {
			sec = sec - 1;
			usec = 1e6 - start->tv_usec;
			usec += stop->tv_usec;
		}
	}

	return sec + (usec * 1e-6);
}

void* checkedCalloc(const size_t count, const size_t size) {
	errno = 0;
	void* result = calloc(count, size);
	if (errno != 0 || result == NULL) {
		perror("Failed to allocate memory.");
		if(result != NULL) {
			free(result);
		}
		exit(1);
	}

	return result;
}

int strToPositiveInteger(const char* str, size_t* value) {
	assert(value != NULL);
	*value = 0;

	if(str == NULL || str[0] == '\0' || str[0] == '-') {
		return 0;
	}

	errno = 0;
	*value = strtoul(str, NULL, 10);
	if(errno != 0 || *value == 0) {
		return 0;
	}
	
	return 1;
}
