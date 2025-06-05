#!/bin/bash
git pull 
mkdir build
cmake -B build
cmake --build build 
./Overlap