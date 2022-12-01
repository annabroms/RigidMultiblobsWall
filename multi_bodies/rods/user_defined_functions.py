'''
Use this module to override forces interactions defined in
multi_body_functions.py. See an example in the file
'''

import multi_bodies_functions
from multi_bodies_functions import *
import autograd.numpy as np
from autograd import grad
from quaternion import Quaternion

#def HGO(r,quaternion_i,quaternion_j):
def HGO(coord):
    #coord on the format [0 r,quaternion_i,quaternion_j
    # set parameters
    r = coord[3:6]
    quaternion_j = coord[6:10]
    quaternion_i = coord[10:-1]
    print("Should double check this implementation")

    a = 0.4
    b = a/10 #perpendicular coeff
    p = 10   #strength of the potential
    chi = (a^2-b^2)/(a^2+b^2)
    s = np.sqrt(2)*b
    epsilon = p*(1-chi^2*np.dot(u1,u2)^2)**(-1/2)
    Ru = r/np.norm(r)

    R = quaternion_i.rotation_matrix()
    u1 = R[:,2]
    R = quaternion_j.rotation_matrix()
    u2 = R[:,2]

    RuU1 = np.dot(Ru,u1)
    RuU2 = np.dot(Ru,u2)

    term2 = (RuU1-RuU2)**2/(1-chi*(np.dot(u1,u2)))))**(-1/2);
    sigma = s*(1-0.5*chi*((RuU1+RuU2)**2/(1+chi*(np.dot(u1,u2)))+term2

    fun = epsilon*np.exp(-np.norm(r)**2/sig**2)
    return fun


def body_body_force_torque_new(r, quaternion_i, quaternion_j, *args, **kwargs):
  '''
  This function returns the external force-torques acting on the bodies.
  It returns an array with shape (2, 3)

  '''
  #Should differentiate wrt x y z and quaternions

  force_torque = np.zeros((2, 3))
  gradient_fun = grad(HGO)
  coord = np.array([0,r,quaternion_i,quaternion_j])
  print(gradient_fun)



  return force_torque
multi_bodies_functions.body_body_force_torque = bodies_external_force_torque_new
