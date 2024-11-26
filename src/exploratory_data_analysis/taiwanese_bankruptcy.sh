#!/bin/bash

i=0
j=3

echo "Starting execution of script..."
while [ $j -lt 97 ]; do
  python taiwanese_bankruptcy.py $i $j
  i=$(($i+3))
  j=$(($j+3))
done
echo "Completed the execution of script."


