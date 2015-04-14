''' Script to calculate equilibrium MSD from a given trajectory (or trajectories) for
the free tetrahedron. Produces a pkl file which can be read by plotting scripts.'''

import argparse
import cPickle
import cProfile
import logging
import numpy as np
import os
import pstats
import StringIO
import sys
sys.path.append('..')


from config_local import DATA_DIR
from quaternion_integrator.quaternion import Quaternion
import tetrahedron_free as tf
from utils import MSDStatistics
from utils import calc_msd_data_from_trajectory
from utils import read_trajectory_from_txt
from utils import StreamToLogger

def calc_tetrahedron_com(location, orientation):
  ''' Function to get tetrahedron center of mass.'''
  r_vectors = tf.get_free_r_vectors(location, orientation)
  center = ((r_vectors[0]*tf.M1 + r_vectors[1]*tf.M2
             + r_vectors[2]*tf.M3 + r_vectors[3]*tf.M4)/
            (tf.M1 + tf.M2 + tf.M3 + tf.M4))

  return center

def calc_tetrahedron_vertex(location, orientation):
  ''' Function to get tetrahedron center of mass.'''
  return np.array(location)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Calculate rotation and '
                                   'translation MSD from a trajectory '
                                   'generated by tetrahedron_free.py. '
                                   'This assumes the data files are named '
                                   'similar to the following: \n '
                                   'free-tetrahedron-trajectory-dt-0.1-N-'
                                   '100000-scheme-RFD-example-name-#.txt\n'
                                   'where # ranges from 1 to n_runs. '
                                   'tetrahedron_free.py uses this '
                                   'convention.')
  parser.add_argument('-scheme', dest='scheme', type=str, default='RFD',
                      help='Scheme of data to analyze.  Options are '
                      'RFD, FIXMAN, or EM.  Defaults to RFD.')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep of runs to analyze.')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps taken in trajectory '
                      'data to be analyzed.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      help='Data name of trajectory runs to be analyzed.')
  parser.add_argument('-n_runs', dest='n_runs', type=int,
                      help='Number of trajectory runs to be analyzed.')
  parser.add_argument('-end', dest='end', type=float,
                      help='How far to analyze MSD (how large of a time window '
                      'to use).  This is in the same time units as dt.')
  parser.add_argument('--out-name', dest='out_name', type=str, default='',
                      help='Optinoal output name to add to the output Pkl '
                      'file for organization.  For example could denote '
                      'analysis using center of mass v. vertex.')
  parser.add_argument('--profile', dest='profile', type=bool, default=False,
                      help='True or False: Do we profile this run or not. '
                      'Defaults to False. Put --profile 1 to profile.')

  args=parser.parse_args()
  if args.profile:
    pr = cProfile.Profile()
    pr.enable()

  # List files here to process.  They must have the same timestep, etc..
  scheme = args.scheme
  dt = args.dt
  end = args.end
  N = args.n_steps
  data_name = args.data_name
  trajectory_length = 150
  
  # Set up logging
  if args.out_name:
    log_filename = './logs/tetrahedron-msd-calculation-dt-%f-N-%d-%s-%s.log' % (
      dt, N, args.data_name, args.out_name)    
  else:
    log_filename = './logs/tetrahedron-msd-calculation-dt-%f-N-%d-%s.log' % (
      dt, N, args.data_name)
  progress_logger = logging.getLogger('Progress Logger')
  progress_logger.setLevel(logging.INFO)
  # Add the log message handler to the logger
  logging.basicConfig(filename=log_filename,
                      level=logging.INFO,
                      filemode='w')
  sl = StreamToLogger(progress_logger, logging.INFO)
  sys.stdout = sl
  sl = StreamToLogger(progress_logger, logging.ERROR)
  sys.stderr = sl

  trajectory_file_names = []
  for k in range(1, args.n_runs+1):
    if data_name:
      trajectory_file_names.append(
        'free-tetrahedron-trajectory-dt-%g-N-%s-scheme-%s-%s-%s.txt' % (
          dt, N, scheme, data_name, k))
    else:
      trajectory_file_names.append(
        'free-tetrahedron-trajectory-dt-%g-N-%s-scheme-%s-%s.txt' % (
          dt, N, scheme, k))

  ##########
  msd_runs = []
  for name in trajectory_file_names:
    data_file_name = os.path.join(tf.DATA_DIR, 'tetrahedron', name)
    # Check correct timestep.
    params, locations, orientations = read_trajectory_from_txt(data_file_name)
    if (abs(float(params['dt']) - dt) > 1e-7):
      raise Exception('Timestep of data does not match specified timestep.')
    if int(params['n_steps']) != N:
      raise Exception('Number of steps in data does not match specified '
                      'Number of steps.')
    
    # Calculate MSD data (just an array of MSD at each time.)
    msd_data = calc_msd_data_from_trajectory(locations, orientations, 
                                             calc_tetrahedron_com, dt, end,
                                             trajectory_length=trajectory_length)
    
    # append to calculate Mean and Std.
    msd_runs.append(msd_data)

  mean_msd = np.mean(np.array(msd_runs), axis=0)
  std_msd = np.std(np.array(msd_runs), axis=0)/np.sqrt(len(trajectory_file_names))
  data_interval = int(end/dt/trajectory_length) + 1
  time = np.arange(0, len(mean_msd))*dt*data_interval

  msd_statistics = MSDStatistics(params)
  msd_statistics.add_run(scheme, dt, [time, mean_msd, std_msd])

  # Save MSD data with pickle.
  if args.out_name:
    msd_data_file_name = os.path.join(
      '.', 'data',
      'tetrahedron-msd-dt-%s-N-%s-end-%s-scheme-%s-runs-%s-%s-%s.pkl' %
    (dt, N, end, scheme, len(trajectory_file_names), data_name, args.out_name))
  else:
    msd_data_file_name = os.path.join(
      '.', 'data',
      'tetrahedron-msd-dt-%s-N-%s-end-%s-scheme-%s-runs-%s-%s.pkl' %
    (dt, N, end, scheme, len(trajectory_file_names), data_name))
    
  with open(msd_data_file_name, 'wb') as f:
    cPickle.dump(msd_statistics, f)
  
  if args.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()  
  
  
    
    
      
    
