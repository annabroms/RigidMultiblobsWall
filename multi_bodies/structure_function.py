
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
    print(N)
    for p in range(numPart):
    #for p in range(1):
        for i in range(N-1):

            temp = np.dot(orientList[i,p,:],orientList[i+1,p,:])
            #print(temp)
            S = S + temp
           # print(S)
    #return S/(numPart*N)
    print(S)
    print("print N %u " % N)
    return S/(numPart*(N-1))


#steps = 5 #number of different runs to collect statistics from. Could be a single
steps = 4
#one with many steps (10^5 at least) as we anyhow subdivide the interval
dtVec = np.logspace(-3,0,steps)
#dtVec = np.logspace(-3,-1,steps)
T = 1 #final simulation time NOT USED
eta = 1
numPart = 10 #number of particles in the simulation
res = 1 # sets resolution for the rods, an intiger 1 2 3 4 with 4 the finest,
save_freq = 1 #save frequency: 1 means that every time step is saved.
ar = 20


mob_r = 14.434758355589102
Dr = mob_r/eta #assuming kbt = 1, double check the scaling with eta


folder = "rods/data/dynamic_rods_T%u_N%u_conc" % (T,numPart)
folder = "rods/data/dynamic_rods_N%u_conc2" % (numPart)
folder = "rods/data/dynamic_rods_N%u_conc_eta1" % (numPart)
#folder = "rods/data/dynamic_rods_N%u_conc" % (numPart)
freqList = range(1,20)
freqList = range(1,30)
freqList = range(1,9)
#freqList = [1,10]
N = 1000
#N = 500
#N = 4

print("WARNING: this version of the function does now work. see the same function name _alt")

config = "random10" #later - we want to loop over configurations here with different concentrations.
# We can do this as a for-loop over different concentrations.
configList = ["random10"]
configList = ["L%1.2f" % (i) for i in [5, 2, 1, 0.5, 0.3]]
configList = ["L%1.2f" % (i) for i in [2]]
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
    for dt in dtVec[0:2]:
        # extract, from the same file, also multiples of the time_step
        print('dt is %f ' % dt)
        for s in freqList:
            #Ns = round(T/(s*dt))
            #print("s is %u" % s)
            Ns = int(N/s) #dont think round is necessary?
            #print(Ns)
            if Ns > 50: #was 100 here
                #N = round(T/dt)

                #read files
                name = "dt%1.5f_eta%1.2f" % (dt,eta)
                print(name)
                fileName = "%s/%s.%s_%s" %(folder,name,config,c)
                orientList = np.zeros(shape=(Ns+1,numPart, 3))
                #loop over all steps
                for i in range(Ns+1):
                    stepName = "%s.%.8u.clones" % (fileName,i*s) #extract time-steps with the specified frequency
                    try:
                        f=open(stepName,"r")
                        print(i*s)
                    except:
                        print('failed to load file')

                    orientList[i,:,:] = readOrientations(f,numPart)
                    #read orientation vector for every particle, with every particle stored with a quaternion
                    f.close()
                #send orientList to computation of the structure quantity for this dt
                print(orientList[i,:,:])

                S[count] = computeStructFun(orientList,numPart,Ns+1)
                dtVecLarge[count] = s*dt
                print("start different time-step")
                count=count+1
            #print(S)

    print(S)
    S = S[np.abs(S)>1e-8]
    # print(S)
    #
    dtVecLarge = dtVecLarge[dtVecLarge>0]
    print(dtVecLarge)
    ind = sorted(range(len(dtVecLarge)), key=lambda k: dtVecLarge[k])
    #print(ind)
    #ind = range(np.size(dtVecLarge))
    plt.figure(2)
    plt.loglog(dtVecLarge[ind],np.arccos(S[ind]),'.-')
    #plt.loglog(dtVecLarge,np.arccos(S),'.-')
    plt.figure(1)
    plt.semilogx(dtVecLarge[ind],S[ind],'.-')
    #plt.semilogx(dtVecLarge,S,'.-')

plt.figure(2)
plt.ylabel('acos S')
plt.xlabel('dt')

plt.figure(1)
plt.ylabel('S')
plt.xlabel('dt')
plt.semilogx(dtVecLarge,np.exp(-2*Dr*dtVecLarge))

plt.show()
