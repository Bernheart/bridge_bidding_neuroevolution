#!/bin/bash

START=$1
END=$2

caffeinate -dimsu bash <<EOF
g++ -O3 -march=native -std=c++17 generator.cpp -I../include -L./dds/src -ldds -L/opt/homebrew/opt/boost/lib -lboost_thread -lboost_system -o generator

for ((i=$START; i<=$END; i++)); do
    echo "batch: \$i"
    ./generator "\$i" > "./data/batch\$i.csv"
done

echo "All batches generated."
EOF