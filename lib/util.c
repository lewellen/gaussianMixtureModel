#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"

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
