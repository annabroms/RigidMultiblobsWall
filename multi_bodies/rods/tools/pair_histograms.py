import numpy as np
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool # for file processing in parallel
#from numpy import quaternion
from scipy.stats import ks_2samp

import sys
sys.path.append('../../../quaternion_integrator')

from quaternion import Quaternion
#from quaternion_integrator.quaternion import Quaternion

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
    """
    This function rotates the second quaternion 'q2' so that the first quaternion 'q1' coincides with [1 0 0 0].

    Parameters:
    q1 (numpy array): A numpy array of shape (4,) representing a unit quaternion (of particle 1)
    q2 (numpy array): A numpy array of shape (4,) representing a unit quaternion (of particle 2)
    x2 (numpy array): A numpy array of shape (3,) representing the 3D coordinate of particle 2

    Returns:
    numpy array: A numpy array of shape (4,) representing the rotated quaternion
    """

    q1 = q1.getAsVector()
    q2 = q2.getAsVector()

    #assert q1.shape == (4,) and q2.shape == (4,), "Inputs should be numpy arrays of shape (4,)"
    assert np.isclose(np.linalg.norm(q1), 1.0) and np.isclose(np.linalg.norm(q2), 1.0), "Inputs should be unit quaternions"

    # Compute the rotation matrix
    R1 = quaternion_to_rotation_matrix(q1)
    x2_new = R1 @ x2
    R2 = rotation_matrix(x2_new[0],x2_new[1])

    # Rotate the second quaternion
    q2_rotated = R2 @ R1 @ q2[1:4]  #make sure rotation works here...
    # Now, the rotation corresponds to the y-axis of the coordinate point correspoinding zero and the quaternion of the first particle coinciding with [1 0 0 0]

    return Quaternion(np.concatenate(([q2[0]],q2_rotated)))



import numpy as np

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
    # convert quaternion to rotation matrix
    R = q.rotation_matrix()
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
    p1,r1 = endpoints(q1, L, x1)
    p2,r2 = endpoints(q2, L, x2)
    return distance(p1, r1, p2, r2)-2*R


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

def getAlpha(x1,x2):
    #translate system
    x2_temp = x2-x1
    return angle_between_vectors([1,0,0],x2_temp)


# Define the process function to apply to each group of three lines
def process_group(lines):
    # Split each line into words and convert them to numbers
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
        return sDist,ccDist,alphaDist,phi,theta
    else:
        return sDist
#place in main file

#test first to rotate q by the matrix generated by q above


#filename = "../../../many_bodyMCMC/run.two.config"
name = "MCMC_analytic_cut5L"
name = "MCMC_analytic_cut2L"
name2 = "MCMC_analytic_cut5L_bdiff"
filename = "../../../../%s.two.config" % name

# for debugging the plotting routine
# name = 'test'
# filename = "../../../many_bodyMCMC/run.two.config"
numSteps = 100000 # number of MCMC runs
#was 10^6 here!
L = 0.5 #particle lenght
R = 0.025
view_all = 1
compare_data = 0
filename2 = "../../../../%s.two.config" % name2

# ccDist = np.empty(numSteps)
# sDist = np.empty(numSteps)
# alphaDist = np.empty(numSteps)
# phiDist = np.empty(numSteps)
# thetaDist = np.empty(numSteps)


# Open the input file and read the lines
with open(filename) as f:
    lines = f.readlines()
# Create a Pool object with the desired number of worker processes
pool = Pool(processes=4)
# Split the input lines into groups of three
groups = [lines[i:i+3] for i in range(0, len(lines), 3)]

# Apply the process_group function to each group of three lines using the Pool object
if view_all:
    #sDist,ccDist,alphaDist,phiDist,thetaDist = pool.map(process_group, groups)
    result = pool.map(process_group, groups)
    print(type(result)) #how many objects in this tuple? 
else:
    sDist = pool.map(process_group, groups)

if compare_data:
    # Now, do the same for another data set
    with open(filename2) as f:
        lines = f.readlines()
        # Create a Pool object with the desired number of worker processes
        pool = Pool(processes=4)
        # Split the input lines into groups of three
        groups = [lines[i:i+3] for i in range(0, len(lines), 3)]

        # Apply the process_group function to each group of three lines using the Pool object
        if view_all:
            sDist2,ccDist2,alphaDist2,phiDist2,thetaDist2 = pool.map(process_group, groups)
        else:
            sDist2 = pool.map(process_group, groups)


    sDist = np.array(sDist)
    sDist2 = np.array(sDist2)


    fig,ax = plt.subplots()
    plt.scatter(np.sort(sDist), np.sort(sDist2))
    # Add labels and legend
    plt.xlabel("Quantiles of Data 1")
    plt.ylabel("Quantiles of Data 2")
    fig.savefig('qq_plot_%s_%s.png' % (name, name2))

    # Perform the Kolmogorov-Smirnov test
    statistic, p_value = ks_2samp(sDist, sDist2)
    # Print the results
    print("KS statistic:", statistic)
    print("p-value:", p_value) #should be less than 0.05

    fig,ax = plt.subplots()
    for i in range(20):
        data1 = np.random.choice(sDist, size=int(0.1*len(sDist)), replace=True)
        data2 = np.random.choice(sDist2, size=int(0.1*len(sDist2)), replace=True)
        p25, p50, p75 = np.quantile(data1, [0.25, 0.5, 0.75])

        p1 = [p25, p50, p75]

        #do the same thing for the second data set
        p25, p50, p75 = np.quantile(data2, [0.25, 0.5, 0.75])

        p2 = [p25, p50, p75]

        plt.plot(p1, p2)

    plt.xlabel("Quantiles of Data 1")
    plt.ylabel("Quantiles of Data 2")
    fig.savefig('qq_plot_bootstrap_%s_%s.png' % (name, name2))

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

#cum_mean = np.empty(int(numSteps))
#cum_var = np.empty(int(numSteps))
# print("Total number of points: %u" % numSteps)
# for i in range(numSteps):
#     cum_mean[i] = np.mean(sDist[0:i+1])
#     cum_var[i] = np.var(sDist[0:i+1])

cum_mean = np.cumsum(sDist[:-1]) / np.arange(1, len(sDist))
# Compute the cumulative variance using numpy.cumsum and numpy.cumsum of squares
sDist = np.array(sDist)
cumulative_sum = np.cumsum(sDist[:-1])
cumulative_sum_sq = np.cumsum(sDist[:-1]**2)
cum_var = (cumulative_sum_sq - cumulative_sum**2 / np.arange(1, numSteps+1)) / np.arange(1, numSteps+1)


# Visualise mean
fig,ax = plt.subplots()
ax.semilogx(range(numSteps),cum_mean)
ax.set_ylabel('Mean shortest distance')
ax.set_xlabel('Number of samples')
fig.savefig('Mean_plot_%s.png' % name)

# Plot also the difference of the mean
fig,ax = plt.subplots()
ax.loglog(range(int(numSteps/10)),np.abs(cum_mean[0:int(numSteps/10)]-cum_mean[-1]))
ax.loglog(range(1,int(numSteps/10)),2/np.sqrt(range(1,int(numSteps/10))),'r.-')
ax.set_ylabel('Error in mean of shortest distance')
ax.set_xlabel('Number of samples')
fig.savefig('Mean_error_plot_%s.png' % name)

# Plot variance
fig,ax = plt.subplots()
ax.semilogx(range(numSteps),cum_var)
ax.set_ylabel('Variance shortest distance')
ax.set_xlabel('Number of samples')
fig.savefig('var_plot_%s.png' % name)


############################################################
#histogram of shortest distances

fig, ax1 = plt.subplots()
print(max(sDist))
print(min(sDist))
ax1.hist(sDist, bins=30, edgecolor='black', color='#1f77b4')

# Set axis labels and title
ax1.set_xlabel('Shortest distance')
ax1.set_ylabel('Frequency')
#ax.set_title('Histogram of Random Data')

# Add grid lines and set background color
ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
ax1.set_axisbelow(True)
ax1.set_facecolor('#e0e0e0')


fig.savefig('Shortest_dist_%s.png' % name)

#########################################################
if view_all:
    # Create histogram of the center-center distances
    fig, ax = plt.subplots()
    print(max(ccDist))
    print(min(ccDist))
    ax.hist(ccDist, bins=30, edgecolor='black', color='#1f77b4')

    # Set axis labels and title
    ax.set_xlabel('Center-center distance')
    ax.set_ylabel('Frequency')
    #ax.set_title('Histogram of Random Data')

    # Add grid lines and set background color
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor('#e0e0e0')

    # Show plot

    fig.savefig('Center_dist.png')


    ###########################################################
    ## VISUALISE DISTRIBUTION IN ALPHA
    fig, ax2 = plt.subplots()
    print(max(alphaDist))
    print(min(alphaDist))
    ax2.hist(alphaDist, bins=30, edgecolor='black', color='#1f77b4')

    # Set axis labels and title
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('Frequency')
    #ax.set_title('Histogram of Random Data')

    # Add grid lines and set background color
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.set_facecolor('#e0e0e0')

    fig.savefig('Alpha_dist.png')


    ###########################################################
    ## VISUALISE DISTRIBUTION IN PHI
    fig, ax = plt.subplots()
    print(max(phiDist))
    print(min(phiDist))
    ax.hist(phiDist, bins=30, edgecolor='black', color='#1f77b4')

    # Set axis labels and title
    ax.set_xlabel('Phi')
    ax.set_ylabel('Frequency')
    #ax.set_title('Histogram of Random Data')

    # Add grid lines and set background color
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor('#e0e0e0')

    # Show plot

    fig.savefig('Phi_dist.png')
    # Show plot


    ##############################################################
    ## VISUALISE DISTRIBUTION IN THETA
    fig, ax = plt.subplots()
    print(max(thetaDist))
    print(min(thetaDist))
    ax.hist(thetaDist, bins=30, edgecolor='black', color='#1f77b4')

    # Set axis labels and title
    ax.set_xlabel('Theta')
    ax.set_ylabel('Frequency')
    #ax.set_title('Histogram of Random Data')

    # Add grid lines and set background color
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor('#e0e0e0')

    # Show plot
    fig.savefig('Theta_dist.png')
