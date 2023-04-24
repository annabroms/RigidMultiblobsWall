
import autograd.numpy as np
from autograd import grad
import sys
#from quaternion_integrator.quaternion import Quaternion


#sys.path.append('../quaternion_integrator')
# print(os.getcwd())
#
sys.path.append('..')
sys.path.append('../..')
# sys.path.append('../../..')
from quaternion_integrator.quaternion import Quaternion

from multi_bodies.rods.tools import pair_histograms as ph
# needed to compute shortest distances


#def HGO(r,quaternion_i,quaternion_j):
def HGO(X):
    # X = x,q
    x = X[0:6]
    q_all = X[6:]
    # set parameters
    r = x[3:6]-x[0:3]

    #L = 0.5 # particle length
    #cut_off = 1.5*L
    a = 0.3 # larger range?
    b = a/5
    L = 2*a
    p = 5
    R = 0 # want to compute shortest distance between line segments

    quat1 = Quaternion(q_all[0:4]) #How to create a quaternion?
    quat2 = Quaternion(q_all[4:])

    # quat1 = np.array(q_all[0:4]) #How to create a quaternion?
    # quat2 = np.array(q_all[4:])
    d = ph.shortestDist(x[0:3],x[3:6],quat1,quat2,L,R)

    cut_off = 1.5*(a+b)
    #cut_off = 0.5*cut_off
    #print(cut_off)
    #if np.linalg.norm(r) < cut_off:
    if d < cut_off:
        #print("distance")
        #print(np.linalg.norm(r))




        #a = 0.5 to start with

        #a = 1

        # decays more rapidly with a smaller a
        #a = 1 #will determine the range for the potential. In initial investigations,
        #a/2 sets the approximate range. a = 1 gives range approx 0.5

        # b = a/8 #perpendicular coeff
        #
        #a = 0.3;

        #b = a/6
        #b = b*1.1
        #p = 0.1  #strength of the potential: will set a time-scale.
        #p = 1

        # More stiff if p is larger. NB: also need to guarantee no overlap
        # If p is larger, dt has to be scaled accordingly

        #What is now the probability of an overlap?
        #p = 1
        chi = (a**2-b**2)/(a**2+b**2)
        s = np.sqrt(2)*b

        Ru = r/np.linalg.norm(r)


        #Rewrite!
        #R = quaternion_i.rotation_matrix()

        q = q_all[0:4]
        R = np.array([[1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
        [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
        [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]])


        u1 = R[:,2]

        #R = quaternion_j.rotation_matrix()


        q = q_all[4:]
        R = np.array([[1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
        [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
        [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]])

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
        #print(epsilon*np.exp((-np.linalg.norm(r)**2)/(sigma**2)))
        return epsilon*np.exp((-np.linalg.norm(r)**2)/(sigma**2))
    else:
        return 1e4 #100 #1e4
        #return p*(np.linalg.norm(r)**2 - cut_off**2)
        #return *(np.linalg.norm(r) - cut_off)**2 #note that this function will not be continuous...

def getGradient(x,q):
    my_grad = grad(HGO)
    X = np.concatenate((x,q))
    return my_grad(X)


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
  X = np.concatenate((x,q))
  return HGO(X)
