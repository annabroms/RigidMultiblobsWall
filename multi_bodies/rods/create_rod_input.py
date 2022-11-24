#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:08:23 2022

@author: annabrom
"""

import numpy as np

#write input files for dynamic simulation with different dt
steps = 5
dtVec = np.logspace(-3,-1,steps)
T = 1 #final simulation time
res = 1 # sets resolution for the rods, an intiger 1 2 3 4 with 4 the finest, 
#will affect the accuracy in a non-trivial manner
eta = 1.0 #viscosity
g = 0.0 #gravity
save_freq = 1 #save frequency: 1 means that every time step is saved. 
ar = 20
if ar==20:
    radList = [0.010838866643485, 0.007616276270953, 0.006207359652491, 0.004963143047909] #provdied ar = 20
radius = radList[res-1]

numPart = 10 #number of particles in the simulation
config = "random10" # start configuration

for dt in dtVec:
    N = round(T/dt)
    print(dt)
    print("%u" % N)
    folder = "dynamic_rods_T%u_N%u" % (T,numPart)
    name = "dt%1.3f" % dt
    str = "%input_s/input_%s.dat" %(folder,name)
    print(str)
    f = open(str, "w")
    
    f.write("# Select integrator\n")
    f.write("scheme \t\t\t\t stochastic_GDC_RFD\n\n")
    #f.write("scheme \t\t\t\t deterministic_forward_euler\n\n")
    
    f.write("# Select implementation to compute M and M*f\n")
    f.write("mobility_blobs_implementation\t\t\t\t python_no_wall\n")
    f.write("mobility_vector_prod_implementation\t\t\t\t numba_no_wall\n\n")
    
    f.write("# Select implementation to compute the blobs-blob interactions\n")
    f.write("blob_blob_force_implementation\t\t\t\t None\n\n")
    
    f.write("body_body_force_torque_implementation\t\t\t\t python\n\n")
    
    f.write("# Set time step, number of steps and save frequency\n")
    f.write("dt\t\t\t\t %f\n" % dt)
    f.write("n_steps\t\t\t\t%u\n" %N)
    f.write("n_save \t\t\t\t %u\n\n"% save_freq)
    
    f.write("domain\t\t\t\t no_wall\n\n")
    
    f.write("# Set fluid viscosity (eta), gravity (g) and blob radius\n")
    f.write("eta\t\t\t\t %f\n" %eta)
    f.write("g\t\t\t\t %f\n" %g) 
    f.write("blob_radius\t\t\t\t %1.16f\n\n" % radius)
    
    f.write("# Set output name\n")
    f.write("output_name\t\t\t\t rods/data/%s/%s\n\n" % (folder,name))
    
    f.write("# Load rigid bodies configuration, provide *.vertex and *.clones files\n")
    f.write("structure rods/Structures/rt_optrod_aspect%u_res%u.vertex rods/Structures/%s.clones\n" %(ar,res,config))
    
    f.close()
  
   # f = open("demofile3.txt", "r")
   # print(f.read())