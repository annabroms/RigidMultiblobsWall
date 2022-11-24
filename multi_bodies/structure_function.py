
# The purpose of this file is to construct the structure function discussed with 
# Fredrik during the fibrils meeting for rod_like particles

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# import scipy
# import sys 
# sys.path.append('../multi_bodies')
# #from multi_bodies import multi_bodies_functions
# import multi_bodies

# import multi_bodies_functions
# from mobility import mobility as mb
from quaternion_integrator.quaternion import Quaternion
# from quaternion_integrator.quaternion_integrator_multi_bodies import QuaternionIntegrator
# from quaternion_integrator.quaternion_integrator_rollers import QuaternionIntegratorRollers
# from body import body
# from read_input import read_input
# from read_input import read_vertex_file
# from read_input import read_clones_file
# from read_input import read_slip_file
# from read_input import read_velocity_file
# from read_input import read_constraints_file
# from read_input import read_vertex_file_list



#from ... import multi_bodies

# Have to run the test for a range of different dt. 
# Store each run in its own folder. 
# Say that we take N time-steps with the smallest dt

def readOrientations(f,numPart):
    #reads quaternions from file and converts result to orientation vectors.  
    f.readline()
    orient = np.zeros(shape=(numPart, 3))
    for i in range(numPart):
        s = f.readline()
        l = s.split() #these are still strings
        l = []
        for t in s.split():
            l.append(float(t))
        q = Quaternion(np.array(l[3:])) #this is the quaternion for the particle. Now, turn it into a direction vector 
        R = q.rotation_matrix()
        u = R[:,2]
        orient[i,:] = u
   # print(orient)    
    return orient
    

def computeStructFun(orientList,numPart,N):
    #Takes in a 3D tensor where the first index corresponds to time-step number, then particle number 
    S = 0
    for p in range(numPart):
        for i in range(N-1):
            S = S + np.dot(orientList[i,p,:],orientList[i+1,p,:])       
           # print(S)
    return S/(numPart*N)


steps = 5
dtVec = np.logspace(-3,-1,steps)
T = 1 #final simulation time
numPart = 10 #number of particles in the simulation
res = 1 # sets resolution for the rods, an intiger 1 2 3 4 with 4 the finest, 
save_freq = 1 #save frequency: 1 means that every time step is saved. 
ar = 20
config = "random10"
folder = "rods/data/dynamic_rods_T%u_N%u" % (T,numPart)
    
S = np.zeros(np.size(dtVec))

count = 0
for dt in dtVec:
    N = round(T/dt)
    #loop over all steps
    #read files
    name = "dt%1.3f" % dt
    fileName = "%s/%s.%s" %(folder,name,config)
    orientList = np.zeros(shape=(N,numPart, 3))
    for i in range(N):
        print("This is step %u" % i)
        stepName = "%s.%.8u.clones" % (fileName,i+1)
        f=open(stepName,"r")
        orientList[i,:,:] = readOrientations(f,numPart)                
        #read orientation vector for every particle, with every particle stored with a quaternion
        f.close()
    #send orientList to computation of the structure quantity for this dt
    S[count] = computeStructFun(orientList,numPart,N)
    print("start different time-step")
    count=count+1 
print(S)         

plt.figure()        
plt.loglog(dtVec,np.arccos(S),'.-')
plt.ylabel('acos S')
plt.xlabel('dt')
plt.show()
print(matplotlib.__file__)
print(matplotlib.__version__)
       # f.open(,"r")
