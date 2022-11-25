
# The purpose of this file is to construct the structure function discussed with 
# Fredrik during the fibrils meeting for rod_like particles

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# import scipy
import sys 
sys.path.append('../quaternion_integrator')
# #from multi_bodies import multi_bodies_functions
# import multi_bodies

# import multi_bodies_functions
# from mobility import mobility as mb
#from quaternion_integrator.quaternion import Quaternion
from quaternion import Quaternion
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


#steps = 5 #number of different runs to collect statistics from. Could be a single 
steps = 1
#one with many steps (10^5 at least) as we anyhow subdivide the interval
dtVec = np.logspace(-3,-1,steps)
T = 1 #final simulation time
numPart = 10 #number of particles in the simulation
res = 1 # sets resolution for the rods, an intiger 1 2 3 4 with 4 the finest, 
save_freq = 1 #save frequency: 1 means that every time step is saved. 
ar = 20
folder = "rods/data/dynamic_rods_T%u_N%u_conc" % (T,numPart)
freqList = range(1,20)


config = "random10" #later - we want to loop over configurations here with different concentrations. 
# We can do this as a for-loop over different concentrations.
configList = ["random10"]
configList = ["L%1.2f_tol001" % (i) for i in [5, 2, 1, 0.5, 0.3]]
#configList = ["L%1.2f_tol001" % (i) for i in [0.5, 0.3]]
#for this few particles it could maybe be interesting with 5 2 1 0.5 0.3 or something like that

plt.figure(1)
plt.clf()

plt.figure(2)
plt.clf()

for c in configList:
    S = np.zeros(np.size(dtVec)*np.size(freqList))
    dtVecLarge = np.zeros(np.size(dtVec)*np.size(freqList))
    count = 0
    #run over simulations with different time-step sizes
    for dt in dtVec:
        # extract, from the same file, also multiples of the time_step
        for s in freqList:
            Ns = round(T/(s*dt))
            if Ns > 100:
                N = round(T/dt)
            
                #read files
                name = "dt%1.3f_%s" % (dt, c)
                fileName = "%s/%s.%s_%s" %(folder,name,config,c)
                orientList = np.zeros(shape=(Ns,numPart, 3))
                #loop over all steps
                for i in range(Ns-1):           
                    stepName = "%s.%.8u.clones" % (fileName,(i+1)*s) #extract time-steps with the specified frequency
                    f=open(stepName,"r")
                    orientList[i,:,:] = readOrientations(f,numPart)                
                    #read orientation vector for every particle, with every particle stored with a quaternion
                    f.close()
                #send orientList to computation of the structure quantity for this dt
                print(Ns)
                S[count] = computeStructFun(orientList,numPart,Ns)
                dtVecLarge[count] = s*dt
                print("start different time-step")
                count=count+1            
             
    S = S[S>0]
    dtVecLarge = dtVecLarge[dtVecLarge>0]
    ind = sorted(range(len(dtVecLarge)), key=lambda k: dtVecLarge[k])   
    plt.figure(2)         
    plt.loglog(dtVecLarge[ind],np.arccos(S[ind]),'.-')
    plt.figure(1)
    plt.semilogx(dtVecLarge[ind],S[ind],'.-')
    
plt.figure(2)    
plt.ylabel('acos S')
plt.xlabel('dt')
plt.show()   

plt.figure(1)    
plt.ylabel('S')
plt.xlabel('dt')
plt.show()


