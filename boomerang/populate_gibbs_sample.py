''' 
Populate files with samples of the boomerang from
the Gibbs Boltzmann distribution.  Can do one wall with gravity and
repulsion, or two walls with just gravity.
'''

import numpy as np
import os
import sys

import boomerang as bm
from utils import write_trajectory_to_txt


if __name__ == '__main__':

  gfactor = float(sys.argv[2])  # multiple of 'earth's gravity'

  original_mass = np.array(bm.M)
  bm.M = original_mass*gfactor
  # Store parameters to write to file, so we know which 
  # parameters this distribution corresponds to.
  params = {'M': bm.M, 'REPULSION_STRENGTH': bm.REPULSION_STRENGTH,
            'ETA': bm.ETA, 'A': bm.A, 'KT': bm.KT,
            'DEBYE_LENGTH': bm.DEBYE_LENGTH,
            'G': gfactor, 'N_BLOBS': len(bm.M)}

  n_samples = int(sys.argv[1])
  trajectory = [[], []]
  for k in range(n_samples):
    sample = bm.generate_boomerang_equilibrium_sample()
    trajectory[0].append(sample[0])
    trajectory[1].append(sample[1].entries)


  file_name = 'boomerang-samples-g-%s.txt' % gfactor
  file_name = os.path.join('.', 'data', file_name)
  write_trajectory_to_txt(file_name, trajectory, params)
  

    


  





