'''
Use this module to override forces interactions defined in
multi_body_functions.py. See an example in the file examples/boomerang_suspensions
'''

# import multi_bodies_functions
# from multi_bodies_functions import *

import sys
sys.path.append('../..')

import multi_bodies._futhark_interaction as _futhark_interaction
#import futhark_tools._futhark_interaction
#import futhark_tools as _futhark_interaction
from futhark_ffi import Futhark


context = Futhark(_futhark_interaction)

import matplotlib.pyplot as plt

# from . import multi_bodies
# from multi_bodies_functions import *

from tools import pair_histograms as ph


import numpy as np

from quaternion_integrator.quaternion import Quaternion
from many_bodyMCMC import potential_pycuda_user_defined as hgo  #contains the definition of the HGO as defined by Anna
#from futhark_tools import futhark_interaction

def calc_body_body_force_torque_new(x1,x2,q1,q2):
#def calc_body_body_force_torque_new(r, quaternion_i, quaternion_j, *args, **kwargs):
  '''
  This function returns the external force-torques acting on the bodies.
  It returns an array with shape (4, 3)

  '''
  #Should differentiate wrt x y z and quaternions

  # x1 = bodies[0].location
  # x2 = bodies[1].location
  # q1 = bodies[0].orientation
  # q2 = bodies[1].orientation

 # print("compute from HGO")

  force_torque = np.zeros((2, 3))
  force_torque_bodies = np.zeros((4, 3))

  grad= hgo.getGradient(np.concatenate((x1,x2)),np.concatenate((q1.getAsVector(),q2.getAsVector())))
  Q1 = q1.gradToTourqueMatrix()
  Q2 = q2.gradToTourqueMatrix()

  # Add forces
  force_torque_bodies[0] = -grad[0:3]
  force_torque_bodies[2] = -grad[3:6]

  # Add torques
  force_torque_bodies[1] = -Q1 @ grad[6:10]
  force_torque_bodies[3] = -Q2 @ grad[10:14]

  return force_torque_bodies


#multi_bodies_functions.calc_body_body_force_torque = calc_bodies_external_force_torque_new
if __name__ == '__main__':

    #generate new valid random configuration by checking shortest distance
    # Want to compare HGO derived with futhark and python
    d = -1
    R = 0.025
    L = 0.5

    q1 = Quaternion([1,0,0,0])
    q2 = Quaternion([1,0,0,0])
    x1 = np.array([0,0,0])

    while (d < 0) or (d > L):
        #generate new random configuration
        x1 = np.random.uniform(-5, 5, 3)
        x2 = np.random.uniform(-5, 5, 3)
        q1.random_orientation()
        q2.random_orientation()

        d = ph.shortestDist(x1,x2,q1,q2,L,R)

    print("Distance is %.2e" % d)
    FT_python = calc_body_body_force_torque_new(x1,x2,q1,q2)
    print("Anna HGO")
    print(FT_python)

    Nbodies = 2
    locations = np.zeros((Nbodies, 3))
    orientations = np.zeros((Nbodies, 4))
    locations[0] = x1
    locations[1] = x2
    orientations[0] = q1.getAsVector()
    orientations[1] = q2.getAsVector()

    #NB - these parameters must be the same as in the potential definition
    epsilon = 10
    sigma_par = 1
    sigma_ort = 1/10
    print(sigma_ort)
    force_torque_bodies = context.hgoInteraction(epsilon, sigma_par, sigma_ort, locations, orientations)
    FT = context.from_futhark(force_torque_bodies)
    print("Futhark HGO")
    print(FT)
    print("Difference in each component")
    print(FT_python-FT)
    print("Relative error")
    #Should normalise with the force for the force and the torque for the torque!
    print((FT_python-FT)/np.linalg.norm(FT))

    print("Difference in torque on second particle")
    print(FT[3]-FT_python[3])


    ode = 0
    if ode:
        case = 3

        # Next, we want to look at time-scales
        if case == 1:
            delta = 0.01
            q1 = Quaternion([1,0,0,0])
            q2 = Quaternion([np.sqrt(2)/2, 0, np.sqrt(2)/2, 0])

            x1 = np.array([0,0,0])
            x2 = np.array([0,0,L/2+R+delta])
        elif case == 2:
            delta = 0.1
            q1 = Quaternion([1,0,0,0])
            # Direction of the second particle is [1 1 0]
            q2 = Quaternion([np.sqrt(2)/2, -1/2, 1/2, 0])
            x1 = np.array([0,0,0])
            x2 = np.array([2*R+delta,0,0])
        else:
            delta = 0.001
            q1 = Quaternion([1,0,0,0])
            q2 = Quaternion([1,0,0,0])

            x1 = np.array([0,0,0])
            x2 = np.array([2*R+delta,0,0])




        #Check the start distance
        p1,r1 = ph.endpoints(q1, L, x1)
        p2,r2 = ph.endpoints(q2, L, x2)
        #d = ph.distance(p1, r1, p2, r2)
        cd = ph.centerDist(x1,x2)
        sd = ph.shortestDist(x1,x2,q1,q2,L,R)
        # print(cd)
        # print(sd)


        #time step the dynamics
        dt = 1e-2 #time step size #Was 1e-3 here before
        N = 1000 # number of time steps
        alphaVec = np.zeros(N)
        thetaVec = np.zeros(N)
        phiVec = np.zeros(N)
        ccVec = np.zeros(N)
        sVec = np.zeros(N)
        potVec = np.zeros(N)
        tVec = np.zeros(N)

        for i in range(N):
            vel = calc_body_body_force_torque_new(x1,x2,q1,q2)
            x1 = x1 + dt*vel[0]
            x2 = x2 + dt*vel[2]
            print(vel)
            quat1_dt = Quaternion.from_rotation(vel[1]*dt)
            q1 = quat1_dt*q1
            quat2_dt = Quaternion.from_rotation(vel[3]*dt)

            q2 = quat2_dt*q2

            #compute angles and distances between the green_particles
            alphaDist = ph.getAlpha(x1,x2)
            r,theta,phi = ph.spherical_q2(q1,q2,x2,L)
            ccDist = ph.centerDist(x1,x2)
            potVal = ph.getPotential(x1,x2,q1.getAsVector(),q2.getAsVector())
            sDist = ph.shortestDist(x1,x2,q1,q2,L,R)

            alphaVec[i] = alphaDist
            thetaVec[i] = theta
            phiVec[i] = phi
            ccVec[i] = ccDist
            sVec[i] = sDist
            potVec[i] = potVal
            tVec[i] = i*dt

        print(ccVec[-1])
        fig,ax = plt.subplots()
        plt.plot(tVec,alphaVec)
        ax.set_ylabel('alpha')
        ax.set_xlabel('t')
        fig.savefig('alpha_scale.png')

        fig,ax = plt.subplots()
        plt.plot(tVec,thetaVec)
        ax.set_ylabel('theta')
        ax.set_xlabel('t')
        fig.savefig('theta_scale.png')

        fig,ax = plt.subplots()
        plt.plot(tVec,phiVec)
        ax.set_ylabel('phi')
        ax.set_xlabel('t')
        fig.savefig('phi_scale.png')

        fig,ax = plt.subplots()
        plt.plot(tVec,ccVec)
        ax.set_ylabel('center dist')
        ax.set_xlabel('t')
        fig.savefig('center_dist_scale.png')

        fig,ax = plt.subplots()
        plt.plot(tVec,sVec)
        ax.set_ylabel('shortes dist')
        ax.set_xlabel('t')
        fig.savefig('shortest_dist_scale.png')

        fig,ax = plt.subplots()
        plt.plot(tVec,potVec)
        ax.set_ylabel('potential')
        ax.set_xlabel('t')
        fig.savefig('potential_scale.png')

    ########################################################
    # #Now, we want to compute the Lyaponov exponent
    #
    # #We have to check if the perturbed system is a valid configuration
    # x1_start,q1_start,x2_start,q2_start = get_initial(case)
    # M = 10 # number of perturbations of the initial condition
    #
    #
    # for epsilon in np.logspace(-3,-1,M):
    #     for k in range(M):
    #         #perturb initial preconditions
    #         d = 0
    #         #Is the configuration valid?
    #         while d<0:
    #
    #             x1 = x1_start + np.random.uniform(-epsilon,epsilon,3)
    #             x2 = x1_start + np.random.uniform(-epsilon,epsilon,3)
    #             #perturebations for the quaternions
    #             pq1 = np.random.uniform(-epsilon,epsilon,3);
    #             pq2 = np.random.uniform(-epsilon,epsilon,3);
    #
    #             quat1_dt = Quaternion.from_rotation(pq1)
    #             q1 = quat1_dt*q1_start
    #             quat2_dt = Quaternion.from_rotation(pq2)
    #             q2 = quat2_dt*q2_start
    #             d = ph.shortestDist(x1,x2,q1,q2,L,R)
