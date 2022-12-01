
# The purpose of this file is to construct the structure function discussed with
# Fredrik during the fibrils meeting for rod_like particles

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
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


#steps = 5 #number of different runs to collect statistics from. Could be a single
steps = 1
#one with many steps (10^5 at least) as we anyhow subdivide the interval
dtVec = np.logspace(-3,0,steps)
#steps = 3
#dtVec = np.logspace(-3,-1,steps)

eta = 1
numPart = 10 #number of particles in the simulation
res = 1 # sets resolution for the rods, an intiger 1 2 3 4 with 4 the finest,
save_freq = 1 #save frequency: 1 means that every time step is saved.
ar = 20

mob_r = 14.434758355589102
#mob_r = 14.434758355589102*8*math.pi
#mob_r = 2*mob_r #testing
Dr = mob_r/eta #assuming kbt = 1, double check the scaling with eta


# folder= "rods/data/dynamic_rods_N%u_conc2" % (numPart)
folder = "rods/data/dynamic_rods_N%u_conc_eta1" % (numPart)
#folder = "rods/data/dynamic_rods_N%u" % (numPart)

#folder = "rods/data/dynamic_rods_N%u_conc" % (numPart)

freqList = range(1,100)
#freqList = range(1,10)
#freqList = range(1,30)
#freqList = [1,10]
N = 1000 #number of steps taken with dt in each file
#N = 500
#N = 4


config = "random%u" % numPart #later - we want to loop over configurations here with different concentrations.
# We can do this as a for-loop over different concentrations.
configList = ["random10"]
configList = ["L%1.2f" % (i) for i in [5, 2, 1, 0.5, 0.3]]
configList = ["L%1.2f" % (i) for i in [2]]
#configList = ["L%1.2f_tol001" % (i) for i in [0.5, 0.3]]freqList = range(1,10)

#for this few particles it could maybe be interesting with 5 2 1 0.5 0.3 or something like that

plt.figure(1)
plt.clf()

plt.figure(2)
plt.clf()

for c in configList:
    #S = np.zeros(np.size(dtVec)*np.size(freqList))
    #S = np.zeros((1,np.size(freqList,0)*np.size(dtVec)))
    print(np.size(freqList))
    print(np.size(dtVec))
    MSAD = np.zeros((1,(np.size(freqList))*np.size(dtVec)))
    S= np.zeros((1,(np.size(freqList))*np.size(dtVec)))

    dtVecLarge = []
    count = 0
    print(np.shape(S))
    #run over simulations with different time-step sizes
    for dt in dtVec:
        # extract, from the same file, also multiples of the time_step
                        #read files
        print(dt)
        name = "dt%1.5f_eta%1.2f" % (dt,eta)
        fileName = "%s/%s.%s_%s" %(folder,name,config,c)

        #loop over the N steps to colleect orientations
        orientList = np.zeros(shape=(N+1,numPart, 3))
        for i in range(N+1):
            stepName = "%s.%.8u.clones" % (fileName,i) #extract time-steps with the specified frequency

            f=open(stepName,"r")
            orientList[i,:,:] = readOrientations(f,numPart)
            #read orientation vector for every particle, with every particle stored with a quaternion
            f.close()

        #now, use the data to collect correlations
        for s in freqList:
            #loop over time-steps
            St = 0
            MS = 0
            for i in range(N-s+1):
                #print(np.reshape(orientList[i,:,:],(numPart*3,1)))
                #print(np.shape(orientList[i,:,:]))
                St = St + np.dot(np.reshape(orientList[i,:,:],(1,numPart*3)),np.reshape(orientList[i+s,:,:],(numPart*3,1)))
                #print(np.shape(np.linalg.norm(orientList[i+s,:,:]-orientList[i,:,:],axis=1)))

                MS = MS + sum(np.linalg.norm(orientList[i+s,:,:]-orientList[i,:,:],axis=1)**2)
            # print("get right index" )
            # print(s-1)
            # print()
            # print(s-1+count*np.size(freqList))
            S[0,s-1+count*np.size(freqList)] = St/((N-s)*numPart)
            MSAD[0,s-1+count*np.size(freqList)] = MS/((N-s)*numPart) #computes mean squared

            print("start different time-step")

            #print(S)
        #print(S)
        count=count+1
        dtVecLarge.append(dt*freqList)

    #S = S[np.abs(S)>1e-8,:]
    dtVecLarge = np.array(dtVecLarge).flatten()
    #print(dtVecLarge)
    #print(S)
    #
    #dtVecLarge = dtVecLarge[dtVecLarge>0]
    #print(dtVecLarge.reshape((1)))
    ind = sorted(range(len(dtVecLarge)), key=lambda k: dtVecLarge[k])
    #print(ind)
    #ind = range(np.size(dtVecLarge))
    plt.figure(2)
    #plt.loglog(dtVecLarge[ind],np.arccos(S[ind]),'.-')
    plt.semilogx(dtVecLarge,np.arccos(S.transpose()),'b.-')
    plt.figure(1)
    #plt.semilogx(dtVecLarge[ind],S[ind],'.-')
    plt.semilogx(dtVecLarge,S.transpose(),'b.-')

    plt.figure(3)
    plt.semilogx(dtVecLarge,MSAD.transpose(),'b.-')
    plt.semilogx(dtVecLarge,2-2*S.transpose(),'r+-')
    plt.semilogx(dtVecLarge,2*np.ones(np.shape(dtVecLarge)),'k--')


plt.figure(2)
plt.ylabel('MAD')
plt.xlabel('dt')
#plt.show()

plt.figure(1)
plt.ylabel('S')
plt.xlabel('dt')
plt.semilogx(dtVecLarge,np.exp(-2*Dr*dtVecLarge))


plt.figure(3)
plt.ylabel('MSAD')
plt.xlabel('dt')

plt.figure(4)
endInd = 10
print(MSAD[:,0:endInd])
print(dtVecLarge[0:endInd])
plt.plot(dtVecLarge[0:endInd],MSAD[:,0:endInd].transpose(),'b.-')
plt.plot(dtVecLarge[0:endInd],4*Dr*dtVecLarge[0:endInd],'c.-')
p = np.polyfit(dtVecLarge[0:endInd],MSAD[:,0:endInd].transpose(),1)
print(p)

plt.figure()
plt.plot(dtVecLarge[0:endInd],dtVecLarge[0:endInd]*Dr)

plt.show()
