#!/bin/bash

version=$1


for y in '2014' '2016' '2019'
do
    echo '****************' start evaluating CROHME $y '****************'
    bash scripts/test/eval.sh $version $y 4
    echo 
done
