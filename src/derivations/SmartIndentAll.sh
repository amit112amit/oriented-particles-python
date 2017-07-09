#!/bin/bash 

for f in ./*txt; do
    ./smartIndent ${f} "${f%.*}_f.txt" 70 '\'
done
