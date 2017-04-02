#ifndef UTIL_H
#define UTIL_H

#include <stdlib.h>

#include <sys/time.h>

double calcElapsedSec(struct timeval* start, struct timeval* stop);
 
void* checkedCalloc(const size_t count, const size_t size);

int strToPositiveInteger(const char* str, size_t* value);

#endif
