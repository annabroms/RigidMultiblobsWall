
# The purpose of this file is to construct the structure function discussed with
# Fredrik during the fibrils meeting for rod_like particles

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import tikzplotlib
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
    #reads quaternions from one file per time-step and converts result to orientation vectors.
    f.readline()
    orient = np.zeros(shape=(numPart, 3))
    for i in range(numPart):
        s = f.readline()
        l = s.split() #these are still strings
        l = []
        count = 0
        for t in s.split():
            l.append(float(t))
            count = count + 1

        print(count)
        q = Quaternion(np.array(l[3:])) #this is the quaternion for the particle. Now, turn it into a direction vector
        R = q.rotation_matrix()
        u = R[:,2]
        #print(np.linalg.norm(u))
        #if np.abs(np.linalg.norm(u)-1)>1e-14:
        #    raise ValueError

        orient[i,:] = u
   # print(orient)
    return orient

def readAllOrientations(f,numSteps,numPart):
    #read orientations from full simulation at once
    orientList = np.zeros(shape=(numSteps+1,numPart, 3))
    for i in range(numSteps+1):
        s = f.readline()
        print(s)
        for k in range(numPart):
            print(k)
            s = f.readline()
            l = []
            for t in s.split():
                l.append(float(t))
            q = Quaternion(np.array(l[3:])) #this is the quaternion for the particle. Now, turn it into a direction vector
            #print(np.array(l[3,:]))
            #print(q)
            R = q.rotation_matrix()
            u = R[:,2]
            #print(u)
            orientList[i,k,:] = u
    return orientList



if __name__ == '__main__':
    #steps = 5 #number of different runs to collect statistics from. Could be a single
    steps = 1
    #one with many steps (10^5 at least) as we anyhow subdivide the interval
    dtVec = np.logspace(-4,0,steps)
    print(dtVec)
    #dtVec = np.logspace(-6,0,steps)
    #steps = 3
    #dtVec = np.logspace(-3,-1,steps)


    eta = 1 #viscosity
    numPart = 1 #number of particles in the simulation
    res = 1 # sets resolution for the rods, an intiger 1 2 3 4 with 4 the finest,
    save_freq = 1 #save frequency: 1 means that every time step is saved.
    ar = 20 # L/R for the particle
    single_file = 1 # Collect data from a single file (or alternatively from one file per time-step)

    mob_r = 14.434758355589102 #for the qbx particle
    mobVec = [14.576095945261537,14.404657971291515, 14.458404941747011,14.430435295093176] #for different resolutions of the multiblob particle
    mob_r = mobVec[res-1]
    Dr = mob_r/eta #assuming kbt = 1


    # folder= "rods/data/dynamic_rods_N%u_conc2" % (numPart)
    folder = "rods/data/dynamic_rods_N%u_conc_eta1" % (numPart)
    folder = "rods/data/dynamic_rods_N%u_one" % (numPart)
    #folder = "rods/data/dynamic_rods_N%u_smallerST" % (numPart)

    #folder = "rods/data/dynamic_rods_N%u_conc" % (numPart)

    #Multiples of the simulation time step to investigate
    freqList = range(1,100)
    #freqList = [range(1,21) 30:10:100 100:100:1000 1000:1000:50000]
    #freqList = [list(range(1,21)) list(range(30,100,10)) list(range(100,1000,100)), list(range(1000,50000,1000))]
    #freqList = [range(1,21), range(30,100,10),range(100,1000,100),range(1000,50000,1000)]
    freqList = []
    for i in range(1,21):
        freqList.append(i)
    for i in range(30,100,10):
        freqList.append(i)
    for i in range(100,1000,100):
        freqList.append(i)
    for i in range(1000,50000,1000):
        freqList.append(i)
    for i in range(1000,90000,1000):
       freqList.append(i)
    print(freqList)

    #freqList.append([30 + 10*k] for k in range(7))
    #freqList = range(1,10)
    #freqList = range(1,30)
    #freqList = [1,10]
    N = 100000 #number of steps taken with dt in each file
    #NN = 150000 #test
    #N = 100
    NN = N
    #NN = N
    #N = 500
    #N = 4
    figName = 'single_smaller_eta'
    #figName = 'single_smaller'

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
            if dt<1e-5:
                name = "dt%1.6f_eta%1.2f" % (dt,eta)
            else:
                name = "dt%1.5f_eta%1.2f" % (dt,eta)
            print(name)
            print(folder)
            fileName = "%s/%s.%s_%s" %(folder,name,config,c)
            if numPart == 1:
                print("Single particle only")
                fileName = "%s/%s.single" %(folder,name)
            if not single_file:
                #loop over the N steps to colleect orientations
                orientList = np.zeros(shape=(NN+1,numPart, 3))
                print(fileName)
                for i in range(NN+1):
                    stepName = "%s.%.8u.clones" % (fileName,i) #extract time-steps with the specified frequency

                    f=open(stepName,"r")
                    orientList[i,:,:] = readOrientations(f,numPart)
                    #read orientation vector for every particle, with every particle stored with a quaternion
                    f.close()
            else:
                #read all orientations simultaneously
                name = "%s/dt%1.5f_%s_eta%1.2f" % (folder,dt,c,eta)
                fullFileName = "%s.%s_%s.config" % (name, config,c)
                if numPart == 1:
                    name = "%s/dt%1.6f_eta%1.2f" % (folder,dt,eta)
                    fullFileName = "%s.single.config" % (name)
                f = open(fullFileName,"r")
                orientList = readAllOrientations(f,N,numPart)
                f.close()
            print(orientList)
            #now, use the data to collect correlations
            k = 1
            for s in freqList:
                #loop over time-steps

                St = 0
                MS = 0
                N = NN-s

                for i in range(N+1):
                    #print(np.reshape(orientList[i,:,:],(numPart*3,1)))
                    #print(np.shape(orientList[i,:,:]))
                    St = St + np.dot(np.reshape(orientList[i,:,:],(1,numPart*3)),np.reshape(orientList[i+s,:,:],(numPart*3,1)))
                    #print(np.shape(np.linalg.norm(orientList[i+s,:,:]-orientList[i,:,:],axis=1)))

                    MS = MS + sum(np.linalg.norm(orientList[i+s,:,:]-orientList[i,:,:],axis=1)**2)

                #S[0,k-1+count*np.size(freqList)] = St/((N-1)*numPart)
                S[0,k-1+count*np.size(freqList)] = St/((N+1)*numPart)
                print(N)
                MSAD[0,k-1+count*np.size(freqList)] = MS/((N+1)*numPart) #computes mean squared
                k = k+1
                print("start different time-step")

                #print(S)
            #print(S)
            count=count+1
            dtVecLarge.append(dt*np.array(freqList))

        #S = S[np.abs(S)>1e-8,:]
        dtVecLarge = np.array(dtVecLarge).flatten()
        #print(dtVecLarge)
        print(S)
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
        #plt.semilogx(dtVecLarge,2-2*S.transpose(),'r+-') #for debugging
        plt.semilogx(dtVecLarge,2*np.ones(np.shape(dtVecLarge)),'k--')


    plt.figure(2)
    plt.ylabel('MAD')
    plt.xlabel('dt')
    plt.semilogx(dtVecLarge,(math.pi/2)*np.ones(np.shape(dtVecLarge)),'k--')
    tikzplotlib.save("rods/figures/MAD%s.tex" %figName)
    #plt.show()

    plt.figure(1)
    plt.ylabel('Orientational correlation function')
    plt.xlabel('dt')
    plt.semilogx(dtVecLarge,np.exp(-2*Dr*dtVecLarge))
    Sp = S[S>0]
    (ind0,ind1) = np.where(S<0) #cannot take log of negative values
    print(ind1)
    if not np.size(ind1):
        endInd = np.size(dtVecLarge)
    else:
        endInd = ind1[0]-2
    print(np.log(S[:,0:endInd]))
    p = np.polyfit(dtVecLarge[0:endInd],np.log(S[:,0:endInd]).transpose(),1)
    print(p)
    errInt = p[1]
    errD = (-p[0]/2-Dr)/Dr
    plt.title("relative error in Dr: %f, in intersept %f" % (errD,errInt))
    tikzplotlib.save("rods/figures/orient%s.tex" %figName)

    plt.figure(3)
    plt.ylabel('MSAD')
    plt.xlabel('dt')
    tikzplotlib.save("rods/figures/MSAD%s.tex" %figName)

    plt.figure(4)
    endInd = 10

    plt.plot(dtVecLarge[0:endInd],MSAD[:,0:endInd].transpose(),'b.-')
    plt.plot(dtVecLarge[0:endInd],4*Dr*dtVecLarge[0:endInd],'c.-')
    p = np.polyfit(dtVecLarge[0:endInd],MSAD[:,0:endInd].transpose(),1)
    print(p)
    print(4*Dr)
    errD = (Dr-p[0]/4)/Dr
    plt.title("relative error in Dr: %f" % errD)
    tikzplotlib.save("rods/figures/short_term%s.tex" %figName)

    # Check that Im looking at sufficiently small dt
    plt.figure(5)
    plt.plot(dtVecLarge[0:endInd],dtVecLarge[0:endInd]*Dr)
    plt.title("sufficiently small dt?")

    plt.show()
