#!/bin/bash

for i in {1..4}; do
  python3 main.py $i &
  sleep 2
done
