cudaLibObjs = $(patsubst lib/%.cu, obj/cuda/%.o, $(wildcard lib/*.cu))
cLibObjs = $(patsubst lib/%.c, obj/c/%.o, $(wildcard lib/*.c))

bins = $(patsubst src/%.c, bin/%, $(wildcard src/*.c))
figs = $(patsubst doc/%.gpi, obj/%.tex, $(wildcard doc/*.gpi))

ccTool = gcc
ccFlags = -O3 -Wall -std=iso9899:1999

# -lm for math
# -rt for real time clock 
# -lpthread for cpu- parallel  code
# -cuda, lcudart -lstdc++ for linking with nvcc output
ccLibs = -L/usr/local/cuda/lib64 -lm -lrt -lpthread -lcuda -lcudart -lstdc++

nvccTool = nvcc
nvccFlags = -O3
nvccLibs = 

.PHONY: all clean
.PRECIOUS: obj/src/%.o obj/%.dat obj/%-summary.dat obj/%.tex obj/%.eps

# -----------------------------------------------------------------------------
# Top Level Targets
# -----------------------------------------------------------------------------

all: $(bins)

clean:
	rm -rf obj
	rm -rf bin

# -----------------------------------------------------------------------------
# Paper Targets
# -----------------------------------------------------------------------------

bin/document.pdf: doc/document.tex $(figs) | bin
	pdflatex --output-directory=bin doc/document.tex
	bibtex bin/document.aux
	pdflatex --output-directory=bin doc/document.tex
	pdflatex --output-directory=bin doc/document.tex

obj/%.tex: obj/%-summary.dat doc/%.gpi | obj
	gnuplot -e "argInput='$<'; argOutput='$@'" $(word 2, $^)

obj/%-summary.dat: analysis/%.py bin/% | obj
	python $< > $@

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

bin/%: obj/src/%.o obj/c-lib.o obj/cuda-lib.o | bin
	$(ccTool) $(ccFlaggs) $< obj/c-lib.o obj/cuda-lib.o -o $@ $(ccLibs) 

obj/src/%.o: src/%.c | obj/src
	$(ccTool) $(ccFlags) -I./include -c $< -o $@ $(ccLibs) 

obj/src: | obj
	mkdir -p ./obj/src

obj:
	mkdir -p ./obj

bin:
	mkdir -p ./bin

# -----------------------------------------------------------------------------
# C Library
# -----------------------------------------------------------------------------

obj/c-lib.o: $(cLibObjs) | obj
	ld -r $(cLibObjs) -o obj/c-lib.o

obj/c/%.o: lib/%.c | obj/c
	$(ccTool) $(ccFlags) -I./include -c $< -o $@ $(ccLibs)

obj/c: | obj
	mkdir -p ./obj/c

# -----------------------------------------------------------------------------
# CUDA Library
# -----------------------------------------------------------------------------

obj/cuda-lib.o: $(cudaLibObjs) | obj
	$(nvccTool) -lib $(cudaLibObjs) -o obj/cuda-lib.o

obj/cuda/%.o: lib/%.cu | obj/cuda
	$(nvccTool) $(nvccFlags) -I./include -c $< -o $@ $(nvccLibs)

obj/cuda: | obj
	mkdir -p ./obj/cuda

