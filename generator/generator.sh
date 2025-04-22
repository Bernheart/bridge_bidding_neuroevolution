#!/bin/bash

BATCH_SIZE=64

if [ $# -eq 1 ]; then
  START=$(($(cat ./data/no_batches$BATCH_SIZE.txt) + 1))
  END=$1
else
  START=$1
  END=$2
fi

caffeinate -dimsu bash <<EOF
g++ -O3 -march=native -std=c++17 generator.cpp -I../include -L./dds/src -ldds -L/opt/homebrew/opt/boost/lib -lboost_thread -lboost_system -o generator

for ((i=$START; i<=$END; i++)); do
    echo "batch: \$i"
    ./generator "\$i" > "./data/$BATCH_SIZE/batch\$i.csv"
    echo "\$i" > "./data/no_batches$BATCH_SIZE.txt"
done

echo "All batches generated."
EOF