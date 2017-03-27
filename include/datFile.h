#ifndef DATFILE_H
#define DATFILE_H

#include <stdlib.h>
#include <stdio.h>

int getFileLength(
	FILE* handle, 
	size_t* outFileLength
	);

char* loadFile(
	const char* filePath, 
	size_t* outFileLength
	);

int isValidDatFile(
	const char* contents, const size_t contentsLength, 
	size_t* outNumLines, size_t* outValuesPerLine
	);
 
double* parseDatFile(
	const char* filePath, 
	size_t* outNumPoints, size_t* outPointDim
	);

#endif
