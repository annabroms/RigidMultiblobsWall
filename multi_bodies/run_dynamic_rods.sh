#!/bin/bash

#for f in rods/dynamic_rods_T1_N10/*.dat;
for f in rods/input_dynamic_rods_T1_N10_conc/*.dat;
do python multi_bodies.py --input-file "$f"; done
