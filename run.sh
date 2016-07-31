#!/bin/bash

while read -r i ; do
	echo $i
	python kmeans-color.py "$i"
done < list.txt
