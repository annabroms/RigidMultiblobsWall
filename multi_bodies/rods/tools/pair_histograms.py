import numpy as np
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool # for file processing in parallel
import multiprocessing
#from numpy import quaternion
from scipy.stats import ks_2samp
import sys
import os
import timeit
from itertools import repeat
from tabulate import tabulate
from scipy import stats


#from ../../../many_bodyMCMC


#
# sys.path.append('../../../quaternion_integrator')

sys.path.append('../../..')
sys.path.append('../..')
from quaternion_integrator.quaternion import Quaternion

from body import body


#from ...quaternion_integrator.quaternion import Quaternion
#from quaternion_integrator.quaternion import Quaternion
# sys.path.append('../../../many_bodyMCMC')
# import potential_pycuda_user_defined
from many_bodyMCMC import potential_pycuda_user_defined
#from many_bodyMCMC import potential_pycuda_user_defined


from futhark_ffi import Futhark
import multi_bodies._futhark_interaction as _futhark_interaction
import multi_bodies.futhark_interaction as fi


def centerDist(x1,x2):
    " Center center distance for the particles"
    return np.linalg.norm(x1-x2)


def quaternion_to_rotation_matrix(q):
    """
    This function returns the rotation matrix that rotates the input unit quaternion 'q' to the quaternion [1 0 0 0].

    Parameters:
    q (numpy array): A numpy array of shape (4,) representing a unit quaternion

    Returns:
    numpy array: A numpy array of shape (3,3) representing the rotation matrix
    """
    q = np.array(q)
    assert q.shape == (4,), "Input should be a numpy array of shape (4,)"
    assert np.isclose(np.linalg.norm(q), 1.0), "Input quaternion should be a unit quaternion"

    R = np.zeros((3,3))
    q0, q1, q2, q3 = q
    R[0,0] = 1 - 2*q2**2 - 2*q3**2
    R[0,1] = 2*q1*q2 - 2*q0*q3
    R[0,2] = 2*q1*q3 + 2*q0*q2
    R[1,0] = 2*q1*q2 + 2*q0*q3
    R[1,1] = 1 - 2*q1**2 - 2*q3**2
    R[1,2] = 2*q2*q3 - 2*q0*q1
    R[2,0] = 2*q1*q3 - 2*q0*q2
    R[2,1] = 2*q2*q3 + 2*q0*q1
    R[2,2] = 1 - 2*q1**2 - 2*q2**2

    return R

def rotation_matrix(x, y):
    angle = np.arctan2(y, x)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0 ],
                                [sin_theta, cos_theta, 0 ], [0, 0, 1]])
    return rotation_matrix

def rotate_quaternion(q1, q2, x2):
    #IT COULD BE THAT THIS FUNCTION IS VERY SLOW

    """
    This function rotates the second quaternion 'q2' so that the first quaternion 'q1' coincides with [1 0 0 0].

    Parameters:
    q1 (numpy array): A numpy array of shape (4,) representing a unit quaternion (of particle 1)
    q2 (numpy array): A numpy array of shape (4,) representing a unit quaternion (of particle 2)
    x2 (numpy array): A numpy array of shape (3,) representing the 3D coordinate of particle 2

    Returns:
    numpy array: A numpy array of shape (4,) representing the rotated quaternion
    """
    # Compute the rotation matrix
    R1 = q1.rotation_matrix()

    q1 = q1.getAsVector()
    q2 = q2.getAsVector()

    #assert q1.shape == (4,) and q2.shape == (4,), "Inputs should be numpy arrays of shape (4,)"
    assert np.isclose(np.linalg.norm(q1), 1.0) and np.isclose(np.linalg.norm(q2), 1.0), "Inputs should be unit quaternions"



    x2_new = R1 @ x2
    R2 = rotation_matrix(x2_new[0],x2_new[1])

    # Rotate the second quaternion
    q2_rotated = R2 @ R1 @ q2[1:4]  #make sure rotation works here...
    # Now, the rotation corresponds to the y-axis of the coordinate point correspoinding zero and the quaternion of the first particle coinciding with [1 0 0 0]

    return Quaternion(np.concatenate(([q2[0]],q2_rotated)))
    #return np.concatenate(([q2[0]],q2_rotated))



def distance(p1, q1, p2, q2):
    """
    Calculate the shortest distance between two line segments in 3D, given their
    end coordinates p1, q1 and p2, q2.

    Parameters:
    -----------
    p1 : array_like
        The 3D coordinates of the first endpoint of the first line segment.
    q1 : array_like
        The 3D coordinates of the second endpoint of the first line segment.
    p2 : array_like
        The 3D coordinates of the first endpoint of the second line segment.
    q2 : array_like
        The 3D coordinates of the second endpoint of the second line segment.

    Returns:
    --------
    dist : float
        The shortest distance between the two line segments.

        The algorithm used in the function is based on the paper "Distance Between Two Finite Line Segments in Three-Dimensional Space" by David Eberly,
        which was published in the journal "Geometric Tools for Computer Graphics" in 1994. The algorithm is known as the "segment-segment distance algorithm"
    """



    # calculate direction vectors for each line segment
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2

    # calculate dot products
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    # calculate denominators
    D = a*c - b**2
    sD = D
    tD = D

    # calculate numerators and parameters for closest points
    if D < 1e-6:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    # calculate closest points
    sc = 0.0 if abs(sN) < 1e-6 else sN / sD
    tc = 0.0 if abs(tN) < 1e-6 else tN / tD
    pc = p1 + sc*u
    qc = p2 + tc*v

    # calculate distance between closest points
    d = pc - qc
    dist = np.sqrt(np.dot(d, d))

    return dist

def endpoints(q, L, c):
    """
    Calculate the end points of a particle of length L described by quaternion q and center coordinate c.

    """

    # convert quaternion to rotation matrix
    R = q.rotation_matrix() #if a quatrernion is used
    #R = quaternion_to_rotation_matrix(q)
    # calculate the direction vector of the line segment
    v = np.array([0, 0, L/2])
    # rotate the direction vector by the quaternion
    v = np.dot(R, v)
    # calculate the endpoint coordinates
    p1 = c + v
    p2 = c - v
    return p1, p2

def shortestDist(x1,x2,q1,q2,L,R):
    #returns shortest distance between the rods
    L = L-2*R #correct for the semi-spherical caps
    p1,r1 = endpoints(q1, L, x1)
    p2,r2 = endpoints(q2, L, x2)

#    print(np.linalg.norm(p1-r1)-L) #this is zero
    # print("End coord")
    # print(p1)
    # print(r1)
    # print(p2)
    # print(r2)
    d = distance(p1, r1, p2, r2)-2*R

    # if d < 0:
    #     print(d)
    #
    #     fig, ax = plt.subplots()
    #
    #     ax = fig.add_subplot(111, projection='3d')
    #     plt.plot([p1[0],r1[0]],[p1[1],r1[1]],[p1[2],r1[2]])
    #     plt.plot([p2[0],r2[0]],[p2[1],r2[1]],[p1[2],r1[2]])
    #     #ax.axis('equal')
    #     plt.show()


    return d


def spherical_coordinates(q, L):
    """
    This function takes a quaternion 'q' and a length 'L' as input and returns the spherical coordinates for the line
    segment specified by the direction of the quaternion 'q' and length 'L'.

    Parameters:
    q (numpy array): A numpy array of shape (4,) representing a unit quaternion that specifies the direction of the line segment
    L (float): A float representing the length of the line segment

    Returns:
    tuple: A tuple of two floats (r, theta) representing the spherical coordinates of the line segment, where 'r' is the
           radial distance and 'theta' is the polar angle in radians.
    """

    q = q.getAsVector()
    assert q.shape == (4,), "Input quaternion should be a numpy array of shape (4,)"
    assert np.isclose(np.linalg.norm(q), 1.0), "Input quaternion should be a unit quaternion"

    # Extract the direction vector from the quaternion
    v = q[1:]

    # Convert the direction vector to spherical coordinates
    r = L


    if np.isclose(np.linalg.norm(v[:2]), 0.0):
        phi = 0.0
        theta = math.pi/2
    else:
        phi = np.arctan2(v[1], v[0])
        theta = np.arccos(v[2] / np.linalg.norm(v))

    return r, theta, phi

def spherical_q2(q1,q2,x2,L):

    q2_rotated = rotate_quaternion(q1, q2, x2)

    #Hmm... should I also rotate here so that the center coordinates are in the same plane, more explicitly?
    return spherical_coordinates(q2_rotated,L)

def angle_between_vectors(a, b):
    """
    This function computes the angle between the vector [0 0 1] and the vector given by the difference between the 3D
    coordinates 'a' and 'b'.

    Parameters:
    a (numpy array): A numpy array of shape (3,) representing a 3D coordinate
    b (numpy array): A numpy array of shape (3,) representing a 3D coordinate

    Returns:
    float: The angle in radians between the vector [0 0 1] and the vector given by the difference between 'a' and 'b'
    """
    a = np.array(a)
    b = np.array(b)
    assert a.shape == (3,) and b.shape == (3,), "Inputs should be numpy arrays of shape (3,)"

    v = b - a  # Compute the vector between 'a' and 'b'
    v /= np.linalg.norm(v)  # Normalize the vector

    angle = np.arccos(np.dot([0, 0, 1], v))  # Compute the angle between [0 0 1] and 'v'

    return angle

def getPotential(x1,x2,q1,q2):
    #Here, it depends on what data we are looking at!
    return potential_pycuda_user_defined.HGO(np.concatenate((x1,x2,q1,q2)))

def getPotentialSingle(x1,x2,q1,q2):
    #Here, it depends on what data we are looking at!
    return potential_pycuda_user_defined.HGO(np.concatenate((x1,x2,q1,q2)).astype(np.float32)) #just to try this!np.concatenate((x1,x2,q1,q2)))

def getBodies(x1,x2,q1,q2):
    Nbodies = 2
    struct_locations = np.zeros((Nbodies, 3))
    struct_orientations = np.zeros((Nbodies, 4))
    struct_locations[0] = x1
    struct_locations[1] = x2
    struct_orientations[0] = q1
    struct_orientations[1] = q2
    struct_ref_config = np.array([[0, 0, 0]])
    bodies = []
    for i in range(2):
        #Create minimalistic description of the bodies
        b = body.Body(struct_locations[i], Quaternion(struct_orientations[i]), struct_ref_config, 0)
        #b.ID = read.structures_ID[ID]
        #body_length = b.calc_body_length()
        #max_body_length = (body_length if body_length > max_body_length else max_body_length)

        b.location_new = b.location_new
        b.orientation_new = b.orientation
        bodies.append(b)
    return np.array(bodies)

def getPotentialFuth(x1,x2,q1,q2):
    #Here, it depends on what data we are looking at!
    bodies = getBodies(x1,x2,q1,q2)
    return fi.compute_total_energy_ref(bodies)

def getPotentialNet(x1,x2,q1,q2,name):
    #Here, it depends on what data we are looking at!
    #name = "../../%s.net" % "hgo5bad"
    name = "../../%s.net" % name
    netparams = fi.read_network_parameter(name)
    bodies = getBodies(x1,x2,q1,q2)
    energy = fi.compute_total_energy_net(bodies, netparams)
    return energy

def getAlpha(x1,x2):
    #translate system
    x2_temp = x2-x1
    return angle_between_vectors([1,0,0],x2_temp)


def getAlignment(x1,x2,q1,q2,cap=float('inf')):
    q = q1.getAsVector()
    R = np.array([[1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
    [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
    [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]])

    u1 = R[:,2]

    q = q2.getAsVector()
    R = np.array([[1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
    [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
    [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]])

    u2 = R[:,2]
    if np.linalg.norm(x1-x2)<cap:
        return np.dot(u1,u2)**2
    else:
        return 0


# Define the process function to apply to each group of three lines
def process_group(lines):
    # Split each line into words and convert them to numbers
#    print(paramname)
    #print(type(args))
    #lines,paramname = args
    numbers1 = [float(x) for x in lines[1].split()]
    numbers2 = [float(x) for x in lines[2].split()]

    x1 = np.array(numbers1[0:3])
    x2 = np.array(numbers2[0:3])

    q1 = Quaternion(np.array(numbers1[3:]))
    q2 = Quaternion(np.array(numbers2[3:]))

    sDist = shortestDist(x1,x2,q1,q2,L,R)

    if view_all:
        alphaDist = getAlpha(x1,x2)
        r,theta,phi = spherical_q2(q1,q2,x2,L)
        ccDist = centerDist(x1,x2)
        potVal = getPotential(x1,x2,q1.getAsVector(),q2.getAsVector())
        potSingle = getPotentialSingle(x1,x2,q1.getAsVector(),q2.getAsVector())
        potRef = getPotentialFuth(x1,x2,q1.getAsVector(),q2.getAsVector())
        potValNetBad = getPotentialNet(x1,x2,q1.getAsVector(),q2.getAsVector(),"hgo5bad")
        potValNetOkay = getPotentialNet(x1,x2,q1.getAsVector(),q2.getAsVector(),"hgo5okay")
        potValNetGood = getPotentialNet(x1,x2,q1.getAsVector(),q2.getAsVector(),"hgo5good")
        alignDist = getAlignment(x1,x2,q1,q2)
        alignDist2 = getAlignment(x1,x2,q1,q2,0.5*L)
        #potVal = getPotential(x1,x2,q1,q2)
        return sDist,ccDist,alphaDist,phi,theta,alignDist,alignDist2,potVal,potSingle,potRef,potValNetBad,potValNetOkay,potValNetGood
    else:
        return sDist

def meanStats(dist,ax,distName,name,col):
    numSteps = len(dist)
    #Computes the cumulative error of the mean for the distribution dist and plots it on axis ax
    cum_mean_dist = np.cumsum(dist[:]) / np.arange(1, len(dist)+1)
    # Compute the cumulative variance using numpy.cumsum and numpy.cumsum of squares
    cumulative_sum = np.cumsum(dist[:])
    cumulative_sum_sq = np.cumsum(dist[:]**2)
    cum_var = (cumulative_sum_sq - cumulative_sum**2 / np.arange(1, len(dist)+1)) / np.arange(1, len(dist)+1)
    ax[0].set_ylabel('Error in mean')
    ax[0].set_xlabel('Number of samples')
    ax[0].grid()
    legend = "%s" % name
    ax[0].loglog(range(int(len(dist)/10)),np.abs(cum_mean_dist[0:int(len(dist)/10)]-cum_mean_dist[-1]),'-',color = col,label=legend)
    #compute the error in a different way
    cum_err =  np.sqrt(cum_var) / np.sqrt(np.arange(1, len(dist)+1))
    # cum_err005 = 0.95*cum_err
    # cum_err001 = 0.99*cum_err

    errvec =np.zeros((1,6))

    #Determine confidence intervals
    alpha = 0.05
    critical_value = stats.norm.ppf(1 - (1 - alpha) / 2)
    standard_err = cum_err[-1]
    margin_of_error = critical_value * standard_err
    errvec[0,0] = cum_mean_dist[-1] - margin_of_error
    errvec[0,1] = cum_mean_dist[-1] + margin_of_error

    alpha = 0.01
    critical_value = stats.norm.ppf(1 - (1 - alpha) / 2)
    margin_of_error = critical_value * standard_err
    errvec[0,2] = cum_mean_dist[-1] - margin_of_error
    errvec[0,3] = cum_mean_dist[-1] + margin_of_error
    errvec[0,4] = standard_err
    errvec[0,5] = np.abs(cum_mean_dist[int(len(dist)/10)]-cum_mean_dist[-1])



    ax[0].loglog(range(int(len(dist))),cum_err[0:int(len(dist))],'--',color = col,label = 'standard error')
    #ax[0].loglog(range(int(len(dist))),cum_err001[0:int(len(dist))],'-.',color = col,label = 'error est alpha=0.01')
    ax[0].set_title(distName)
    ax[0].loglog(range(1,int(numSteps/10)),2/np.sqrt(range(1,int(numSteps/10))),'k.-',label = "O(1/sqrt(N))")
    ax[0].legend()

    # Visualise the mean itself

    ax[1].semilogx(range(numSteps),cum_mean_dist,label=legend,color = col)
    ax[1].set_ylabel('Mean')
    ax[1].set_xlabel('Number of samples')
    ax[1].legend()
    ax[1].grid()


    return np.array(errvec)

    # #Visualise variance
    # ax.semilogx(range(numSteps),cum_var)
    # ax.set_ylabel('Variance shortest distance')
    # ax.set_xlabel('Number of samples')



def readDists(filename):
    with open(filename) as f:
        lines = f.readlines()
    # Create a Pool object with the desired number of worker processes
    pool = Pool(processes=4)
    # Split the input lines into groups of three
    groups = [lines[i:i+3] for i in range(0, len(lines), 3)]

    # Apply the process_group function to each group of three lines using the Pool object
    if view_all:
        #sDist,ccDist,alphaDist,phiDist,thetaDist = pool.map(process_group, groups)
        #paramList = [paramname for i in range(len(groups))]

        # groups = (groups,repeat(paramname))
        # lines,paramname = groups
        #
        #
        result = pool.map(process_group, groups)

        result
        #a_results = pool.apply_async(process_group, args=(groups, paramname))
        #result = pool.starmap(process_group, [groups, paramname])
        # retrieve the return value results
        #results = a_results.get()

        return zip(*result)
    else:
        sDist = pool.map(process_group, groups)
        return sDist

def showHist(dist,ax, name,bins=30):
    #ax.hist(dist, bins=bins, density=True, edgecolor='black', color='#1f77b4',label = name)

    y, bin_edges = np.histogram(dist, bins=bins,density=True)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

    ax.errorbar(
        bin_centers,
        y,
        yerr = 0,
        marker = '.',
        drawstyle = 'steps-mid',
        label = name
    )
    ax.legend()

    ax.set_ylabel('pdf')
    # Add grid lines and set background color
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor('#e0e0e0')
    ax.set_ylim(bottom=0)



def quantilePlot(dist1,dist2,name1,name2,distname):
    fig,ax = plt.subplots()
    for i in range(30):
        data1 = np.random.choice(dist1, size=int(0.1*len(dist1)), replace=True)
        data2 = np.random.choice(dist2, size=int(0.1*len(dist2)), replace=True)
        p25, p50, p75 = np.quantile(data1, [0.25, 0.5, 0.75])

        p1 = [p25, p50, p75]

        #do the same thing for the second data set
        p25, p50, p75 = np.quantile(data2, [0.25, 0.5, 0.75])

        p2 = [p25, p50, p75]

        plt.plot(p1, p2,'r.-')
    plt.plot(p1,p1,'k-')
    #plt.scatter(np.sort(sDist), np.sort(sDist2))

    plt.xlabel("Quantiles of Data 1")
    plt.ylabel("Quantiles of Data 2")

    # Perform the Kolmogorov-Smirnov test
    statistic, p_value = ks_2samp(dist1, dist2)
    # Print the results
    print("KS statistic:", statistic)
    print("p-value:", p_value) #should be less than 0.05
    textstr = "Kolmogorov-Smirnov, KS statistic: %f p_value: %f" % (statistic,p_value)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top')
    ax.set_title("Quantiles with bootstrap: %s\n Data1: %s and Data2: %s" % (distname,name1,name2))
    fig.savefig('qq_plot_bootstrap_dist%s_%s_%s.png' % (distname,name1, name2))





#test first to rotate q by the matrix generated by q above

if __name__ == '__main__':
    #filename = "../../../many_bodyMCMC/run.two.config"

    name1 = "MCMC_ref_cut1.5"
    #name = "MALA_analytic_cut2"
    name2 = "MCMC_bad_cut1.5"
    name3 = "MCMC_okay_cut1.5"
    name4 = "MCMC_good_cut1.5"
    compName = "HGO5"
    #compName = "testing_hgo5"

    #paramNames = ["hgo5bad","hgo5okay","hgo5good"]
    # name = "Langevin_analytic_cut5L_test"
    # name = "MALA_analytic_cut5L_1e-2"
    # name2 = "MALA_analytic_cut5L_1e-1"
    #
    # name = "EM_analytic_cut5L_1e-2"
    # name2 = "EM_analytic_cut5L_1e-1"
    #name = "EM_analytic_cut5L"
    #name2 = "LM_analytic_cut5L"
    #name = "MCMC_analytic_cut5L"
    #name = "LM_analytic_cut5L_1e-1"


    #name = "MCMC_analytic_cut1.5L"
    #name2 = "MCMC_analytic_cut5L_bdiff"
    #name2 = "MCMC_analytic_cut2L"

    # for debugging the plotting routine
    # name = 'test'
    # filename = "../../../many_bodyMCMC/run.two.config"
    #numSteps = 1000000 # number of MCMC runs
#    numSteps = 100000 # number of MCMC runs
    #was 10^6 here!
    L = 0.5 #particle length
    L = 0.3*2
    R = 0.025
    R = 0
    view_all = 1
    compare_data = 1

    names  = ["test"]
    names = [name1, name2]
    #names = ["test", "test"]
    names = [name1]
    test_names = names
    numFiles = len(names)
    #names = [name1, name2, name3, name4]
    #names = [name1,name2]
    print(names)

    #Initialise figures
    sstring = ""
    fig1,ax1 = plt.subplots(2)
    fig2,ax2 = plt.subplots(2)
    fig3,ax3 = plt.subplots(2)
    fig4,ax4 = plt.subplots()
    fig5,ax5= plt.subplots()
    fig6,ax6 = plt.subplots()
    fig7,ax7= plt.subplots()
    fig8,ax8 = plt.subplots()
    fig9,ax9= plt.subplots()
    fig10,ax10= plt.subplots(2)
    fig11,ax11= plt.subplots(2)
    fig12,ax12= plt.subplots()
    fig13,ax13= plt.subplots()

    #store error estimates
    RMSE = np.zeros((4,len(names)))
    RMSE_rel = np.zeros((4,len(names)))

    serr_vec = np.zeros((len(names),6))
    cerr_vec = np.zeros((len(names),6))
    perr_vec = np.zeros((len(names),6))
    aerr_vec = np.zeros((len(names),6))
    acaperr_vec = np.zeros((len(names),6))

    # Store KS tests
    ks_sdist = np.zeros((len(names),2))
    ks_cdist  = np.zeros((len(names),2))
    ks_adist  = np.zeros((len(names),2))
    ks_alphadist  = np.zeros((len(names),2))
    ks_acapdist  = np.zeros((len(names),2))
    ks_tdist = np.zeros((len(names),2))
    ks_phdist  = np.zeros((len(names),2))
    ks_pdist  = np.zeros((len(names),2))
    ks_pNetdist  = np.zeros((len(names),2))


    cols = ['r','b','g','m','c']

    #For each run, do:

    for i,name in enumerate(names):
        filename = "../../../../%s.two.config" % name
        #paramname = "../../%s.net" % paramNames[i+1]
        ####### PREPARE DATA ########################
        # Enables qq-plots
        if i == 1:
            sDist2 = sDist
            ccDist2 = ccDist
            alphaDist2 = alphaDist
            phiDist2 = phiDist
            thetaDist2 = thetaDist
            potDist2 = potDist
            alignDist2 = alignDist
            alignDistCap2 = alignDistCap


        #unpack
        sDist,ccDist,alphaDist,phiDist,thetaDist,alignDist,alignDistCap,potDist,potDistSingle,potDistRef,potDistNetBad,potDistNetOkay,potDistNetGood = readDists(filename)
        # lists = [sDist,ccDist,alphaDist,phiDist,thetaDist,alignDist,alignDistCap,potDist,potDistNetBad,potDistNetOkay,potDistNetGood]
        # for list in lists:
        #     list = np.array(list)
        #     print(type(list))

        sDist = np.array(sDist)
        ccDist = np.array(ccDist)
        alphaDist = np.array(alphaDist)
        phiDist = np.array(phiDist)
        alignDist = np.array(alignDist)
        alignDistCap = np.array(alignDistCap)
        potDist = np.array(potDist)
        potDistSingle = np.array(potDistSingle)
        potDistRef = np.array(potDistRef)
        potDistNetBad = np.array(potDistNetBad)
        potDistNetOkay = np.array(potDistNetOkay)
        potDistNetGood = np.array(potDistNetGood)

        #do stuff with the data

        #Compare netwórk data to analytic potential with ks-test
        if i > 0:
            ks_sdist[i,:]  = np.array(ks_2samp(sDist, sDist2))
            ks_cdist[i,:]  = np.array(ks_2samp(ccDist, ccDist2))
            ks_alphadist[i,:]  = np.array(ks_2samp(alphaDist, alphaDist2))
            ks_adist[i,:]  = np.array(ks_2samp(alignDist, alignDist2))
            ks_acapdist[i,:]  = np.array(ks_2samp(alignDistCap, alignDistCap2))
            ks_phdist[i,:]  = np.array(ks_2samp(phiDist, phiDist2))
            ks_tdist[i,:]  = np.array(ks_2samp(thetaDist, thetaDist2))
            ks_pdist[i,:]  = np.array(ks_2samp(potDist, potDist2))
            if i ==1:
                ks_pNetdist[i,:] = np.array(ks_2samp(potDist, potDistNetBad))
            elif i == 2:
                ks_pNetdist[i,:] = np.array(ks_2samp(potDist, potDistNetOkay))
            elif i == 3:
                ks_pNetdist[i,:] = np.array(ks_2samp(potDist, potDistNetGood))
            else:
                ks_pNetdist[i,:] = np.array(ks_2samp(potDist, potDistRef))


        #compute rms error for the potential, absolute and relative

        MSE = np.square(np.subtract(potDist,potDistRef)).mean()
        RMSE[0,i] = math.sqrt(MSE)
        print(MSE)
        RMSEnorm = np.square(potDist).mean()
        RMSE_rel[0,i] = math.sqrt(MSE)/RMSEnorm

        MSE = np.square(np.subtract(potDist,potDistNetBad)).mean()
        RMSE[1,i] = math.sqrt(MSE)
        RMSEnorm = np.square(potDist).mean()
        RMSE_rel[1,i] = math.sqrt(MSE)/RMSEnorm

        MSE = np.square(np.subtract(potDist,potDistNetOkay)).mean()
        RMSE[2,i] = math.sqrt(MSE)
        RMSEnorm = np.square(potDist).mean()
        RMSE_rel[2,i] = math.sqrt(MSE)/RMSEnorm

        MSE = np.square(np.subtract(potDist,potDistNetGood)).mean()
        RMSE[3,i] = math.sqrt(MSE)
        RMSEnorm = np.square(potDist).mean()
        RMSE_rel[3,i] = math.sqrt(MSE)/RMSEnorm




        #Visualise mean, convergence of the mean and compute statistical errors for different quantities

        print("Starting to visualise...%u" % i)
        print(type(sDist))
        #Convergence of the mean:
        serr_vec[i,:] = meanStats(sDist,ax1,"shortest distance", name,cols[i])
        #cerr005,cerr001,cerr = meanStats(ccDist,ax2,"center-center distance", name,cols[i])
        cerr_vec[i,:] = meanStats(ccDist,ax2,"center-center distance", name,cols[i])
        # perr005,perr001,perr = meanStats(potDist,ax3,"Energy",name,cols[i])
        perr_vec[i,:] = meanStats(potDist,ax3,"Energy",name,cols[i])

        aerr_vec[i,:] = meanStats(alignDist,ax10,"alignment", name,cols[i])
        acaperr_vec[i,:] = meanStats(alignDistCap,ax11,"near alignment", name,cols[i])


        # Histograms over distances (shortest and center distance), the three angles, alpha, phi, theta,
        #and the near and total alignment
        bins = np.linspace(0, 0.5*L, num=61)
        showHist(sDist,ax4,name,bins=bins)
        ax4.set_xlabel('shortest distance')
        sstring = "%s Max shortest distance %s= %f\n " % (sstring, name, np.max(sDist))
        ax4.text(0.05, 0.95, sstring, transform=ax4.transAxes,
                fontsize=12, verticalalignment='top')


        bins = np.linspace(0, 2*L, num=31)
        showHist(ccDist,ax5,name,bins=bins)
        ax5.set_xlabel('center-center distance')

        showHist(alphaDist,ax6,name,bins=30)
        ax6.set_xlabel('alpha')

        showHist(phiDist,ax7,name,bins=30)
        ax7.set_xlabel('phi')

        showHist(thetaDist,ax8,name,bins=30)
        ax8.set_xlabel('theta')

        bins = np.linspace(0, 0.3, num=31)
        showHist(potDist,ax9,name,bins=bins)
        ax9.set_xlabel('U')

        showHist(alignDist,ax12,name,bins=30)
        ax12.set_xlabel('alignment')

        bins = np.linspace(0, 0.2*L, num=31)
        showHist(alignDistCap,ax13,name,bins=bins)
        ax13.set_xlabel('near alignment')

    # Write all figures to file
    fig1.savefig('%s/Mean_err_shortest_dist_%s.png' % (compName,compName))
    fig2.savefig('%s/Mean_err_center_dist_%s.png' % (compName,compName))
    fig3.savefig('%s/Mean_err_potential_%s.png' % (compName,compName))
    fig10.savefig('%s/Mean_err_alignment_%s.png' % (compName,compName))
    fig11.savefig('%s/Mean_err_near_alignment_%s.png' % (compName,compName))

    fig4.savefig('%s/Hist_shortest_dist_%s.png' % (compName,compName))
    fig5.savefig('%s/Hist_center_dist_%s.png' % (compName,compName))
    fig6.savefig('%s/Hist_alpha_%s.png' % (compName,compName))
    fig7.savefig('%s/Hist_phi_dist_%s.png' % (compName,compName))
    fig8.savefig('%s/Hist_theta_dist_%s.png' % (compName,compName))
    fig9.savefig('%s/Hist_U_%s.png' % (compName,compName))
    fig12.savefig('%s/Hist_alignment_%s.png' % (compName,compName))
    fig13.savefig('%s/Hist_near_alignment_%s.png' % (compName,compName))
    plt.close('all')


    # Write statistical errors to table
    stat_names = ["shortest distance", "center-center distance", "alignment", "close alignment","potential value"]
    errvecs = [serr_vec,cerr_vec, perr_vec, aerr_vec, acaperr_vec]

    # Size of statistical errors?
    standard_err = np.zeros((len(test_names),len(stat_names)))
    est_err = np.zeros((len(test_names),len(stat_names)))
    for i,(name, errvec) in enumerate(zip(stat_names, errvecs)):
        print("\nStatistical error estimates: %s" % name)
        standard_err[:,i] = errvec[:,4]
        est_err[:,i] = errvec[:,5]
        print(tabulate(errvec[:,4:], headers=['standard err', 'estimate']))

    #Write to latex
    #Column names
    column_names = stat_names
    # Row names
    row_names = test_names
    # Convert the matrix to a nested list
    matrix_list = standard_err.tolist()
    #matrix_list = [['{:.2e}'.format(element) for element in row] for row in standard_err]
    # Insert row names as the first column in the matrix list
    matrix_list = [[row_name] + row for row_name, row in zip(row_names, matrix_list)]

    # Generate the LaTeX table with column and row names
    latex_table = tabulate(matrix_list, headers=column_names, tablefmt="latex")

    matrix_list = est_err.tolist()
    matrix_list = [[row_name] + row for row_name, row in zip(row_names, matrix_list)]
    # Generate the LaTeX table with column and row names
    latex_table = latex_table + '\n\n'+ tabulate(matrix_list, headers=column_names, tablefmt="latex")

    # Save the LaTeX table to a file
    with open("%s/statistical_errors.tex" % compName, "w") as f:
        f.write(latex_table)

    #Draw confidence intervals
    x_values = list(range(1,len(errvecs)+1))
    xtick_labels = ['shortest distance', 'center-center distance', 'potential','alignment','near alignment']  # Custom xtick labels

    def plot_conf(inds,alpha):
        # Plotting the intervals,
        #fig14,ax14 = plt.subplots()

        for k in range(len(errvecs)):
            errvec = errvecs[k]
            for i in range(numFiles): #assume we have 4 different potentials to compare
                C_curr = errvec[i,:]
                if i==0:
                    c = 'b'
                else:
                    c = 'r'
                plt.errorbar(x_values[k]+0.1*i,np.mean(C_curr[inds]),
                yerr = np.array([[C_curr[inds[0]]],[C_curr[inds[1]]]]),
                fmt='o', capsize=5,color = c)

        # Additional plot settings
        plt.xlabel('Groups')
        plt.ylabel('Values')
        plt.title('Paired Confidence Intervals, alpha =%1.2e' % alpha)

        # Customize xticks
        plt.xticks(x_values, xtick_labels,rotation='45')  # Set xtick locations and labels
        plt.grid(True)

    fig14,ax14 = plt.subplots()
    plt.gcf().set_size_inches(8, 8)  # Set the figure size (increase as needed)
    #plt.tight_layout()  # Adjust the layout to prevent overlap
    plot_conf([0,1],0.05)
    fig14.savefig('%s/Confidence_intervals_0.05_%s.png' % (compName,compName))

    fig15,ax15 = plt.subplots()
    plt.gcf().set_size_inches(8, 8)  # Set the figure size (increase as needed)
    plot_conf([2,3],0.01)
    fig15.savefig('%s/Confidence_intervals_0.01_%s.png' % (compName,compName))









    #Print RMSE
    print("\nPotential RMSE")
    print(tabulate(RMSE, headers=['futhark','bad', 'okay', 'good']))
    print("\nPotential relative RMSE")
    print(tabulate(RMSE_rel, headers=['futhark','bad', 'okay', 'good']))


    #Print KS tests
    names = ["shortest distance", "center-center distance", "alignment", "near alignment", "alpha", "theta", "phi", "potential", "net potential"]
    kslists = [ks_sdist, ks_cdist, ks_adist, ks_acapdist, ks_alphadist, ks_tdist, ks_phdist, ks_pdist, ks_pNetdist]
    pmat005 = np.zeros((numFiles,len(names)),dtype=bool)
    pmat001 = np.zeros((numFiles,len(names)),dtype=bool)
    for i,(name, kslist) in enumerate(zip(names, kslists)):
        print("\nKS test: %s" % name)
        print(tabulate(kslist, headers=['KS', 'p']))
        pmat005[:,i] = [bool(k<0.05) for k in kslist[:,1]]# bool(kslist[:,1]<0.05)
        pmat001[:,i] = [bool(k<0.01) for k in kslist[:,1]]


    # Visualise p values in matrix plot (imshow)
    # Sample row and column labels
    row_labels = ['Futhark','Bad', 'Okay', 'Good']
    row_labels = ['test','test']
    col_labels = names

    # Plotting the boolean matrices
    fig16,ax16 = plt.subplots()
    plt.imshow(pmat005, interpolation='nearest')
    #print(range(pmat.shape[0]))
    plt.xticks(range(pmat005.shape[1]), col_labels,rotation='65')
    plt.yticks(range(pmat005.shape[0]), row_labels)
    # Add gridlines
    #plt.grid(True, color='black', linewidth=0.5)
    fig16.savefig('%s/KS_test_alpha%1.2e%s.png' % (compName,0.05,compName))

    fig17,ax17 = plt.subplots()
    plt.imshow(pmat001, interpolation='nearest')
    #print(range(pmat.shape[0]))
    plt.xticks(range(pmat001.shape[1]), col_labels,rotation='65')
    plt.yticks(range(pmat001.shape[0]), row_labels)
    # Add gridlines
    #plt.grid(True, color='black', linewidth=0.5)
    fig16.savefig('%s/KS_test_alpha%1.2e%s.png' % (compName,0.05,compName))


    # Maybe we want a qq-plot?
    name1 = 'Analytic'
    name2 = 'bad'
    quantilePlot(potDist,potDistNetGood,name1,name2,"potentials")



    # Open the input file and read the lines


    #IF WE DON*T READ THE FILE IN PARALLEL:
        # for i, line in enumerate(file):
        #     l =[]
        #     if i % 3 == 1:
        #         # Extract coodinates for particle 1
        #         for t in line.split():
        #             l.append(float(t))
        #         q1 = Quaternion(np.array(l[3:])) #this is the quaternion for the particle. Now, turn it into a direction vector
        #         #q1 = quaternion(l[3:]) #this is the quaternion for the particle. Now, turn it into a direction vector
        #         x1 = np.array(l[0:3])
        #     elif i % 3 == 2:
        #         #extract coordinate and quaternion for particle 2
        #         for t in line.split():
        #             l.append(float(t))
        #         q2 = Quaternion(np.array(l[3:])) #this is the quaternion for the particle. Now, turn it into a direction vector
        #         #q2 = quaternion(l[3:])
        #         x2 = np.array(l[0:3])
        #     elif i>2:
        #         #compute statistics using these coordinates
        #         if view_all:
        #             sDist[int(i/3-1)] = shortestDist(x1,x2,q1,q2,L,R)
        #         else:
        #             ccDist[int(i/3-1)] = centerDist(x1,x2)
        #             alphaDist[int(i/3-1)] = getAlpha(x1,x2)
        #             r,theta,phi = spherical_q2(q1,q2,x2,L)
        #             phiDist[int(i/3-1)] = phi
        #             thetaDist[int(i/3-1)] = theta


 # TO LOOK AT THE CUTOFF:
    num_nets = 5
    fig1,ax1 = plt.subplots(num_nets,1,figsize=(10, 8))
    fig2,ax2 = plt.subplots(num_nets,1,figsize=(10, 8))
    fig3,ax3 = plt.subplots(num_nets,1,figsize=(10, 8))
    fig4,ax4 = plt.subplots(num_nets,1,figsize=(10, 8))
    fig5,ax5 = plt.subplots(num_nets,1,figsize=(10, 9))
    fig6,ax6 = plt.subplots(num_nets,1,figsize=(10, 10))

    endInd = min(len(sDist),1000)
    #potDist = potDistRef
    for k,(dist,name) in enumerate(zip([potDistNetBad,potDistNetOkay,potDistNetGood,potDistRef,potDistSingle],["Bad","Okay","Good","HGO futhark", "Single precision"])):

        ax1[k].scatter(sDist[0:endInd],potDist[0:endInd],marker='o', facecolors='none', edgecolors='blue',label = "analytic")
        ax1[k].scatter(sDist[0:endInd],dist[0:endInd],marker='o', facecolors='none', edgecolors='orange',label = name)
        print(sum(dist[0:endInd]<0))
        ax2[k].scatter(sDist[0:endInd],potDist[0:endInd],marker='o', facecolors='none', edgecolors='blue',label = "analytic")
        ax2[k].scatter(sDist[0:endInd],dist[0:endInd],marker='o', facecolors='none', edgecolors='orange',label = name)

        ax3[k].scatter(potDist[0:endInd],dist[0:endInd],marker='o', facecolors='none', edgecolors='orange',label = name)
        ax4[k].scatter(potDist[0:endInd],dist[0:endInd],marker='o', facecolors='none', edgecolors='orange',label = name)

        #Visualise the relative error
        ax5[k].loglog(potDist[0:endInd],np.abs(potDist[0:endInd]-dist[0:endInd])/np.abs(potDist[0:endInd]),'.',label = name)
        threshold = 1e-4;
        #p = potDist[0:endInd]
        p = potDist
        indices = np.where(p > threshold)[0]
        ax6[k].loglog(potDist[indices],np.abs(potDist[indices]-dist[indices])/np.abs(potDist[indices]),'.',label = name)
        ax6[k].set_ylim(1e-7, 100)
        yticks = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3,1e-2,1e-1,1,10]
        ax6[k].set_yticks(yticks)
        ax6[k].grid(which='minor', axis='y', linestyle='dotted')
        ax6[k].grid(which='minor', axis='x', linestyle='dotted')


        ax1[k].set_yscale('log')
        ax3[k].set_yscale('log')
        ax3[k].set_xscale('log')

        ax1[k].set_ylabel('U')
        if k ==3:
            ax1[k].set_xlabel('shortest distance')
            ax2[k].set_xlabel('shortest distance')
            ax3[k].set_xlabel('U analytic')
            ax4[k].set_xlabel('U analytic')
        ax1[k].grid()
        ax1[k].legend()

        ax2[k].set_ylabel('U')
        ax2[k].grid()
        ax2[k].legend()


        ax3[k].set_ylabel('U net')
        ax3[k].grid()
        ax3[k].legend()
        ax3[k].loglog(np.logspace(-10,1),np.logspace(-10,1),'k--')


        ax4[k].set_ylabel('U net')
        ax4[k].grid()
        ax4[k].plot(np.linspace(0,2.5),np.linspace(0,2.5),'k--')
        ax4[k].legend()

        ax5[k].grid()
        ax5[k].legend()
        ax5[k].set_ylabel('Relative error in potential')
        ax5[k].set_xlabel('Potential value')

        ax6[k].grid()
        ax6[k].legend()
        ax6[k].set_ylabel('Relative error in potential')
        ax6[k].set_xlabel('Potential value')




    fig1.savefig('%s/Potential_distance_log_%s.png' % (compName,compName))
    fig2.savefig('%s/Potential_distance_%s.png' % (compName,compName))


    fig3.savefig('%s/Potential_potential_log_%s.png' % (compName,compName))
    fig4.savefig('%s/Potential_potential%s.png' % (compName,compName))

    fig5.savefig('%s/Potential_potential_error_%s.png' % (compName,compName))
    fig6.savefig('%s/Potential_potential_error_zoom_%s.png' % (compName,compName))



    print("Finalised plotting")
