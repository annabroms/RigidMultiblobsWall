#!/bin/bash

for f in rods/dynamic_rods_T1_N10/*.dat;
do python multi_bodies.py --input-file "$f"; done
