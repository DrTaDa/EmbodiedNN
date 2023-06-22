#!/bin/bash

for i in {3..5}; do
  /usr/local/opt/python@3.10/bin/python3 main.py $i &
  sleep 2
done
