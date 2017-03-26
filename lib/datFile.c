#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "datFile.h"
#include "util.h"

int getFileLength(FILE* handle, size_t* outFileLength) {
	if(fseek(handle, 0, SEEK_END) != 0) {
		perror("Failed to seek to end of file.");
	} else if ((*outFileLength = ftell(handle)) <= 0) {
		perror("Zero or negative file length encountered.");
	} else if (fseek(handle, 0, SEEK_SET) != 0) {
		perror("Failed to seek to start of file.");
	} else {
		return 1;
	}

	return 0;
}

char* loadFile(const char* filePath, size_t* outFileLength) {
	char* contents = NULL;

	FILE* handle = fopen(filePath, "rb");
	if(handle == NULL) {
		perror(filePath);
	} else {
		if(getFileLength(handle, outFileLength)) {
			contents = (char*)checkedCalloc(*outFileLength + 1, sizeof(char));
			assert(contents != NULL);

			size_t bytesRead = fread(contents, sizeof(char), *outFileLength, handle);
			if (bytesRead != *outFileLength) {
				perror("Number of bytes read does not match number of bytes expected.");
				if (contents != NULL) {
					free(contents);
					contents = NULL;
				}
			} else {
				contents[*outFileLength] = '\0';
			}
		}

		fclose(handle);
		handle = NULL;
	}

	return contents;
}

int isValidDatFile(const char* contents, const size_t fileLength, size_t* numLines, size_t* valuesPerLine) {
	assert(contents != NULL);
	assert(numLines != NULL);
	assert(valuesPerLine != NULL);

	*numLines = 0;
	*valuesPerLine = 0;

	size_t maxValuesPerLine = 0;
	size_t lastNewLine = 0;
	for(size_t i = 0; i < fileLength; ++i) {
		switch(contents[i]) {
			case '#': {
				// Ignore comments
				while(i < fileLength && contents[i] != '\n') 
					++i;
				*valuesPerLine = 0;
				break;
			}
			case '\t': {
				++(*valuesPerLine);
				break;
			}
			case '\n': {
				++(*valuesPerLine);

				if(maxValuesPerLine == 0) {
					maxValuesPerLine = *valuesPerLine;
				} else if (*valuesPerLine != maxValuesPerLine) {
					fprintf(stdout, "%.64s", &contents[lastNewLine]);
					fprintf(stdout, "Line: %zu\n", *numLines);
					fprintf(stdout, "Expect each line to have same number of values. %zu != %zu.\n", *valuesPerLine, maxValuesPerLine);
					return 0;
				}

				lastNewLine = i;
				*valuesPerLine = 0;
				++(*numLines);
				break;
			}
			default: {
				break;
			}
		}
	}

	if(*numLines == 0) {
		return 0;
	}

	*valuesPerLine = maxValuesPerLine;
	return 1;
}

double* parseDatFile(const char* filePath, size_t* numPoints, size_t* pointDim) {
	assert(filePath != NULL);
	assert(numPoints != NULL);
	assert(pointDim != NULL);

	size_t fileLength = 0;
	char* contents = loadFile(filePath, &fileLength);
	if(contents == NULL) {
		return NULL;
	}

	if(!isValidDatFile(contents, fileLength, numPoints, pointDim)) {
		free(contents);
		contents = NULL;
		return NULL;
	}

	double* data = (double*)checkedCalloc(*numPoints * *pointDim, sizeof(double));

	size_t currentPoint = 0;
	for(size_t i = 0, j = 0; i < fileLength; ++i) {
		switch(contents[i]) {
			case '#': {
				// Ignore comments
				while(i < fileLength && contents[i] != '\n') 
					++i;
				j = i;
				break;
			}
			case '\t': 
			case '\n': {
				errno = 0;
				data[currentPoint] = strtod(&contents[j], NULL);
				if(errno != 0) {
					perror("String to double error.");
					free(data);
					free(contents);
					return NULL;
				}

				++currentPoint;
				j = i;
				break;
			}
			default: {
				break;
			}
		}
	}

	for(size_t i = 0; i < currentPoint; ++i) {
		assert( data[i] == data[i] );
		assert( data[i] != -INFINITY );
		assert( data[i] != +INFINITY );
	}

	if(contents != NULL) {
		free(contents);
		contents = NULL;
	}

	return data;
}
