import numpy as np
import sys
from quaternion_integrator.quaternion import Quaternion


#def HGO(r,quaternion_i,quaternion_j):
def HGO(x,q):
    #coord on the format [0 r,quaternion_i,quaternion_j
    # set parameters
    r = x[3:6]-x[0:3]
    #print("distance")
    #print(np.linalg.norm(r))
    quaternion_j = Quaternion(q[0:4]) #How to create a quaternion?
    quaternion_i = Quaternion(q[4:])

    #a = 0.5 to start with
    a = 0.5
    b = a/10 #perpendicular coeff
    p = 100  #strength of the potential
    chi = (a**2-b**2)/(a**2+b**2)
    s = np.sqrt(2)*b

    Ru = r/np.linalg.norm(r)

    R = quaternion_i.rotation_matrix()
    u1 = R[:,2]
    R = quaternion_j.rotation_matrix()
    u2 = R[:,2]
    epsilon = p*(1-chi**2*np.dot(u1,u2)**2)**(-1/2)

    RuU1 = np.dot(Ru,u1)
    RuU2 = np.dot(Ru,u2)

    term2 = (RuU1-RuU2)**2/(1-chi*(np.dot(u1,u2)))
    sigma = s*(1-0.5*chi*((RuU1+RuU2)**2/(1+chi*(np.dot(u1,u2)))+term2))**(-1/2)
    # print(chi)
    # print(sigma)
    # print(-np.linalg.norm(r)**2/(sigma**2))
    # print(np.linalg.norm(r)**2)
    return epsilon*np.exp((-np.linalg.norm(r)**2)/(sigma**2))


def compute_total_energy(bodies, r_vectors, *args, **kwargs):
  '''
  This function compute the energy of the bodies.
  '''

  # Determine number of threads and blocks for the GPU
  number_of_bodies = np.int32(len(bodies))
  # Create location and orientation arrays
  x = np.empty(3 * number_of_bodies)
  q = np.empty(4 * number_of_bodies)
  for k, b in enumerate(bodies):
    x[k*3 : (k+1)*3] = b.location_new
    q[k*4] = b.orientation_new.s
    q[k*4 + 1 : k*4 + 4] = b.orientation_new.p

  #NB: So far we consider only two particles!
  return HGO(x,q)
