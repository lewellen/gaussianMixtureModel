#ifndef DATFILE_H
#define DATFILE_H

#include <stdlib.h>
#include <stdio.h>

int getFileLength(FILE* handle, size_t* outFileLength);

char* loadFile(const char* filePath, size_t* outFileLength); 

int isValidDatFile(const char* contents, const size_t fileLength, size_t* numLines, size_t* valuesPerLine); 

double* parseDatFile(const char* filePath, size_t* numPoints, size_t* pointDim);
 
#endif
