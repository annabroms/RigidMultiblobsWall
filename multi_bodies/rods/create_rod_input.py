#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:08:23 2022

@author: annabrom
"""

import numpy as np
import os

#write input files for dynamic simulation with different dt
steps = 6
dtVec = np.logspace(-5,0,steps)
#dtVec = np.logspace(-2,-1,steps)
#T = 1 #NOT IN USE: final simulation time
timeSteps = 1000  # number of timesteps
res = 1 # sets resolution for the rods, an intiger 1 2 3 4 with 4 the finest,
#will affect the accuracy in a non-trivial manner
eta = 1.0 #viscosity
#eta = 10 #just for testing
g = 0.0 #gravity
save_freq = 1 #save frequency: 1 means that every time step is saved.
ar = 20
if ar==20:
    radList = [0.010838866643485, 0.007616276270953, 0.006207359652491, 0.004963143047909] #provdied ar = 20
radius = radList[res-1]

#numPart = 10 #number of particles in the simulation
numPart = 1
impl = "numba"

#configList = ["random%u_L%1.2f_tol001" % (numPart,i) for i in [5, 2, 1, 0.5, 0.3]] # start configurations of different concenterations
concList = [5, 2, 1, 0.5, 0.3]
concList = [2]

#old folder names
#folder = "dynamic_rods_T%u_N%u_conc" % (T,numPart)
#folder = "dynamic_rods_T%u_N%u_testcuda" % (T,numPart)
#folder = "dynamic_rods_T%u_N%u_movie" % (T,numPart)

folder = "dynamic_rods_N%u_conc" % (numPart)
folder = "dynamic_rods_N%u" % numPart
folder = "dynamic_rods_N%u_1000" % numPart
#folder = "dynamic_rods_N%u_conc" % numPart
path = "input_%s" %folder
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new input directory is created!")

dataPath = "data/%s" % folder
isExist = os.path.exists(dataPath)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(dataPath)
   print("The new data directory is created!")

for c in concList:
    config = "random%u_L%1.2f" % (numPart,c)
    for dt in dtVec:
        name = "dt%1.5f_eta%1.2f" % (dt,eta)
        if numPart == 1:
            str = "%s/input_%s.dat" %(path,name)
        else:
            str = "%s/input_%s_L%1.2f.dat" %(path,name,c)
        print(str)

        #N = round(T/dt)
        N = timeSteps
        print(dt)
        print("%u" % N)

        f = open(str, "w")

        f.write("# Select integrator\n")
#        f.write("scheme \t\t\t\t stochastic_GDC_RFD\n\n")
        f.write("scheme \t\t\t\t stochastic_first_order_RFD_dense_algebra\n\n")
        #f.write("scheme \t\t\t\t deterministic_forward_euler\n\n")

        f.write("# Select implementation to compute M and M*f\n")
        f.write("mobility_blobs_implementation\t\t\t\t python_no_wall\n")
        f.write("mobility_vector_prod_implementation\t\t\t\t %s_no_wall\n\n" % impl)

        f.write("# Select implementation to compute the blobs-blob interactions\n")
        f.write("blob_blob_force_implementation\t\t\t\t None\n\n")

        f.write("body_body_force_torque_implementation\t\t\t\t None \n\n") # change here if we have a force /torque

        f.write("# Set time step, number of steps and save frequency\n")
        f.write("dt\t\t\t\t %f\n" % dt)
        f.write("n_steps\t\t\t\t%u\n" %N)
        f.write("n_save \t\t\t\t %u\n"% save_freq)
        #f.write("save_clones \t\t\t\t one_file\n\n")

        f.write("domain\t\t\t\t no_wall\n\n")

        f.write("# Set fluid viscosity (eta), gravity (g) and blob radius\n")
        f.write("eta\t\t\t\t %f\n" %eta)
        f.write("g\t\t\t\t %f\n" %g)
        f.write("blob_radius\t\t\t\t %1.16f\n\n" % radius)

        f.write("# Set output name\n")
        f.write("output_name\t\t\t\t rods/data/%s/%s\n\n" % (folder,name))

        f.write("# Load rigid bodies configuration, provide *.vertex and *.clones files\n")
        if numPart == 1:
            f.write("structure rods/Structures/rt_optrod_aspect%u_res%u.vertex rods/Structures/single.clones\n" %(ar,res))
        else:
            f.write("structure rods/Structures/rt_optrod_aspect%u_res%u.vertex rods/Structures/%s.clones\n" %(ar,res,config))

        f.close()

   # f = open("demofile3.txt", "r")
   # print(f.read())
