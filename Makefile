libObjs = $(patsubst lib/%.c, obj/lib/%.o, $(wildcard lib/*.c))
bins = $(patsubst src/%.c, bin/%, $(wildcard src/*.c))
figs = $(patsubst doc/%.gpi, obj/%.tex, $(wildcard doc/*.gpi))

ccTool = gcc
ccFlags = -O3 -Wall -std=iso9899:1999
ccLibs = -lm

.PHONY: all clean
.PRECIOUS: obj/src/%.o obj/%.dat obj/%-summary.dat obj/%.tex obj/%.eps

all: $(bins)


bin/document.pdf: doc/document.tex $(figs) | bin
	pdflatex --output-directory=bin doc/document.tex
	bibtex bin/document.aux
	pdflatex --output-directory=bin doc/document.tex
	pdflatex --output-directory=bin doc/document.tex

obj/%.tex: obj/%-summary.dat doc/%.gpi | obj
	gnuplot -e "argInput='$<'; argOutput='$@'" $(word 2, $^)

obj/%-summary.dat: analysis/%.py obj/%.dat | obj
	python $< $(word 2, $^) > $@

obj/%.dat: bin/% | obj
	$< > $@



bin/%: obj/src/%.o | bin
	$(ccTool) $(ccFlaggs) $< -o $@ $(ccLibs) 

bin:
	mkdir -p ./bin

obj/lib.o: $(libObjs)
	ld -r $(libObjs) -o obj/lib.o

obj/src/%.o: src/%.c | obj/src
	$(ccTool) $(ccFlags) -I./include -c $< -o $@ $(ccLibs) 

obj/lib/%.o: lib/%.cc | obj/lib
	$(ccTool) $(ccFlags) -I./include -c $< -o $@ $(ccLibs)
 
obj/src: | obj
	mkdir -p ./obj/src

obj/lib: | obj
	mkdir -p ./obj/lib

obj:
	mkdir -p ./obj

clean:
	rm -rf obj
	rm -rf bin
