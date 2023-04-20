import numpy as np
import sys
import imp
import os.path
from functools import partial

from quaternion_integrator.quaternion import Quaternion

import multi_bodies._futhark_interaction as _futhark_interaction
from futhark_ffi import Futhark

context = Futhark(_futhark_interaction)

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
    potential = context.networkPotential(networkparameter, location, orientation)
    return potential
        
#r_vectors not used in calculation
def calc_body_body_forces_torques_futhark_hgo(bodies, r_vectors, epsilon, sigma_par, sigma_ort, *args, **kwargs):
    Nbodies = len(bodies)
    locations = np.zeros(Nbodies, 3)
    orientations = np.zeros(Nbodies, 4)
    for i in range(Nbodies):
        locations[i] = bodies[i].location
        orientations[i] = bodies[i].orientation.getAsVector()
    force_torque_bodies = context.hgoInteraction(epsilon, sigma_par, sigma_ort, locations, orientations)
    return context.fromFuthark(force_torque_bodies)

def calc_body_body_forces_torques_futhark_net(bodies, r_vectors, networkparameter, *args, **kwargs):
    Nbodies = len(bodies)
    locations = np.zeros(Nbodies, 3)
    orientations = np.zeros(Nbodies, 4)
    for i in range(Nbodies):
        locations[i] = bodies[i].location
        orientations[i] = bodies[i].orientation.getAsVector()
    force_torque_bodies = context.networkInteraction(networkparameter, locations, orientations)
    return context.fromFuthark(force_torque_bodies)

