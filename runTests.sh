#!/bin/bash

make

for file in bin/test*; do
	$file
done
