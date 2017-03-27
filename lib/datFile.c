#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "datFile.h"
#include "util.h"

int getFileLength(FILE* handle, size_t* outFileLength) {
	assert(handle != NULL);
	assert(outFileLength != NULL);

	*outFileLength = 0;
	
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
	assert(filePath != NULL);
	assert(outFileLength != NULL);

	*outFileLength = 0;

	char* contents = NULL;
	FILE* handle = fopen(filePath, "rb");
	if(handle == NULL) {
		perror(filePath);
	} else {
		if(getFileLength(handle, outFileLength)) {
			contents = (char*)checkedCalloc(*outFileLength + 1, sizeof(char));

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

int isValidDatFile(
	const char* contents, const size_t contentsLength, 
	size_t* outNumLines, size_t* outValuesPerLine
) {
	assert(contents != NULL);
	assert(outNumLines != NULL);
	assert(outValuesPerLine != NULL);

	*outNumLines = 0;
	*outValuesPerLine = 0;

	size_t maxValuesPerLine = 0;
	for(size_t i = 0; i < contentsLength; ++i) {
		switch(contents[i]) {
			case '#': {
				// Ignore comments
				while(i < contentsLength && contents[i] != '\n') 
					++i;
				*outValuesPerLine = 0;
				break;
			}
			case '\t': {
				++(*outValuesPerLine);
				break;
			}
			case '\n': {
				++(*outValuesPerLine);

				if(maxValuesPerLine == 0) {
					maxValuesPerLine = *outValuesPerLine;
				} else if (*outValuesPerLine != maxValuesPerLine) {
					fprintf(stdout, "Expect each line to have same number of values. %zu != %zu.\n", *outValuesPerLine, maxValuesPerLine);
					return 0;
				}

				*outValuesPerLine = 0;
				++(*outNumLines);
				break;
			}
			default: {
				break;
			}
		}
	}

	if(*outNumLines == 0) {
		return 0;
	}

	*outValuesPerLine = maxValuesPerLine;
	return 1;
}

double* parseDatFile(const char* filePath, size_t* outNumPoints, size_t* outPointDim) {
	assert(filePath != NULL);
	assert(outNumPoints != NULL);
	assert(outPointDim != NULL);

	size_t contentsLength = 0;
	char* contents = loadFile(filePath, &contentsLength);
	if(contents == NULL) {
		return NULL;
	}

	if(!isValidDatFile(contents, contentsLength, outNumPoints, outPointDim)) {
		free(contents);
		contents = NULL;
		return NULL;
	}

	double* data = (double*)checkedCalloc(*outNumPoints * *outPointDim, sizeof(double));

	size_t currentPoint = 0;
	for(size_t i = 0, j = 0; i < contentsLength; ++i) {
		switch(contents[i]) {
			case '#': {
				// Ignore comments
				while(i < contentsLength && contents[i] != '\n') 
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
