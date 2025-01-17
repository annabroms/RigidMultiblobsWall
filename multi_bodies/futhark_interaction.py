import numpy as np
import sys

import os.path
from functools import partial

sys.path.append('../../..')
sys.path.append('..')
sys.path.append('../..')
from quaternion_integrator.quaternion import Quaternion
from multi_bodies.rods.tools import pair_histograms as ph
#import multi_bodies
#import multi_bodies._futhark_interaction as _futhark_interaction
#from multi_bodies import *
#import _futhark_interaction as _futhark_interaction
import _futhark_interaction as _futhark_interaction
#import futhark_tools._futhark_interaction
#import futhark_tools as _futhark_interaction
from futhark_ffi import Futhark


context = Futhark(_futhark_interaction)

#def read_network_parameter(filename):
def read_network_parameter(filename):
    with open(filename, 'rb') as file:
        parameter = context.restore_networkParameter(file.read())
    return parameter

#quaternions are taken as arrays, this can be changed
def futhark_hgo(location, orientation, epsilon, sigma_par, sigma_ort):
    potential = context.hgoPotential(epsilon, sigma_par, sigma_ort, location, orientation)
    return potential

#networkparamter needs to be read from a file
def futhark_net(location, orientation, networkparameter):
    print(location)
    print(orientation)
    potential = context.networkPotential(networkparameter, location, orientation)
    #potential = context.networkPotential(networkparameter, orientation, location)
    return potential
    #return context.from_futhark(potential)


#r_vectors not used in calculation
def calc_body_body_forces_torques_futhark_hgo(bodies, r_vectors, epsilon, sigma_par, sigma_ort, *args, **kwargs):
    Nbodies = len(bodies)
    locations = np.zeros((Nbodies, 3))
    orientations = np.zeros((Nbodies, 4))
    for i in range(Nbodies):
        locations[i] = bodies[i].location
        orientations[i] = bodies[i].orientation.getAsVector()
    force_torque_bodies = context.hgoInteraction(epsilon, sigma_par, sigma_ort, locations, orientations)
    return context.from_futhark(force_torque_bodies)

def calc_body_body_forces_torques_futhark_net(bodies, r_vectors, networkparameter, *args, **kwargs):
    Nbodies = len(bodies)
    locations = np.zeros((Nbodies, 3))
    orientations = np.zeros((Nbodies, 4))
    for i in range(Nbodies):
        locations[i] = bodies[i].location
        orientations[i] = bodies[i].orientation.getAsVector()
    force_torque_bodies = context.networkInteraction(networkparameter, locations, orientations)
    return context.from_futhark(force_torque_bodies)

def compute_total_energy(bodies, networkparameter):
  '''
  This function compute the energy of the bodies.
  '''

  # Determine number of threads and blocks for the GPU
  Nbodies = np.int32(len(bodies))
  # Create location and orientation arrays

  x = np.zeros((Nbodies, 3))
  q = np.zeros((Nbodies, 4))
  # for k, b in enumerate(bodies):
  #   x[k] = b.location_new
  #   q[k] = b.orientation_new.getAsVector()

  for i in range(Nbodies):
      x[i] = bodies[i].location
      q[i] = bodies[i].orientation.getAsVector()
    #q[k*4 + 1 : k*4 + 4] = b.orientation_new.p
  print(np.shape(x))
  #NB: So far we consider only two particles!
  energy = futhark_net(x,q,networkparameter)
  # epsilon = 1
  # sigma_par = 2
  # sigma_ort = 3
  # energy = futhark_hgo(x, q, epsilon, sigma_par, sigma_ort)
  print(energy)
  return energy

if __name__ == '__main__':

    #generate new valid random configuration by checking shortest distance
    d = -1
    R = 0.025
    L = 0.5

    L = 6
    R = 0.5

    q1 = Quaternion([1,0,0,0])
    q2 = Quaternion([1,0,0,0])

    while d < 0 or (d > 0.5*R):
        #generate new random configuration
        x1 = np.random.uniform(-5, 5, 3)
        x2 = np.random.uniform(-5, 5, 3)
        q1.random_orientation()
        q2.random_orientation()


        d = ph.shortestDist(x1,x2,q1,q2,L,R)


    print("Distance is %d " % d)

    #FT = calc_body_body_force_torque_new(x1,x2,q1,q2)
    Nbodies = 2
    locations = np.zeros((Nbodies, 3))
    orientations = np.zeros((Nbodies, 4))
    locations[0] = x1
    locations[1] = x2
    orientations[0] = q1.getAsVector()
    orientations[1] = q2.getAsVector()
    epsilon = 1
    sigma_par = 2
    sigma_ort = 3
    filename = 'p1_a2_b3.net'
    force_torque_bodies = context.hgoInteraction(epsilon, sigma_par, sigma_ort, locations, orientations)
    FT = context.from_futhark(force_torque_bodies)
    parameter = read_network_parameter(filename)
    force_torque_bodies = context.networkInteraction(parameter, locations, orientations)
    FT_net = context.from_futhark(force_torque_bodies)
    print(FT[0])
    print(FT_net[0])

    #print(FT[0])
