
import numpy as np
import time
import sys
import subprocess
import os.path

sys.path.append('..')

from multi_bodies.rods.tools import pair_histograms as ph

try:
      import pickle as cpickle
except:
    try:
        import cpickle
    except:
        import _pickle as cpickle

# Find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
    try:
        #import many_body_potential_pycuda
        #from many_bodyMCMC import many_body_potential_pycuda
        import potential_pycuda_user_defined
        from body import body
        from quaternion_integrator.quaternion import Quaternion
        from read_input import read_input
        from read_input import read_vertex_file, read_clones_file
        import general_application_utils as utils

        found_functions = True
    except ImportError:
        path_to_append += '../'
        print('searching functions in path ', path_to_append)
        sys.path.append(path_to_append)
        if len(path_to_append) > 21:
              print('\nProjected functions not found. Edit path in many_body_MCMC.py')
              sys.exit()

def get_blobs_r_vectors(bodies, Nblobs):
  '''
  Return coordinates of all the blobs with shape (Nblobs, 3).
  '''
  r_vectors = np.empty((Nblobs, 3))
  offset = 0
  for b in bodies:
      num_blobs = b.Nblobs
      r_vectors[offset:(offset+num_blobs)] = b.get_r_vectors()
      offset += num_blobs
      return r_vectors


def set_blob_potential(implementation):

    '''
    Set the function to compute the blob-blob potential
    to the right function.
    '''
    if implementation == 'None':
        def default_zero_r_vectors(*args, **kwargs):
            return 0
        return default_zero
    elif implementation == 'python':
        return calc_blob_potential_python
    elif implementation == 'C++':
        return calc_blob_potential_boost
    elif implementation == 'pycuda':
        return many_body_potential_pycuda.calc_blob_potential_pycuda


if __name__ == '__main__':
    # script takes input file as command line argument or default 'data.main'
    if len(sys.argv) != 2:
        input_file = 'data.main'
    else:
        input_file = sys.argv[1]

    # Read input file
    read = read_input.ReadInput(input_file)

    # Copy input file to output
    subprocess.call(["cp", input_file, read.output_name + '.inputfile'])

    # Set random generator state
    if read.random_state is not None:
        with open(read.random_state, 'rb') as f:
            np.random.set_state(cpickle.load(f))
    elif read.seed is not None:
        np.random.seed(int(read.seed))

    # Save random generator state
    with open(read.output_name + '.random_state', 'wb') as f:
        cpickle.dump(np.random.get_state(), f)

    # Parameters from the input file
    blob_radius = read.blob_radius
    periodic_length = read.periodic_length
    dt = 4e-2 #time-step: read from file instead
    dt = 1e-2
    weight = 1.0 * read.g
    kT = read.kT
    MCMC = 1
    LM = 0
    L = 0.5
    R = 0.025
    sample_r_vectors = []

    # Create rigid bodies
    bodies = []
    body_types = []
    max_body_length = 0.0
    for ID, structure in enumerate(read.structures):
        print('Creating structures = ', structure[1])
        struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
        num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])
        body_types.append(num_bodies_struct)
    # Creat each body of type structure
        for i in range(num_bodies_struct):
            b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, blob_radius)
            b.ID = read.structures_ID[ID]
            body_length = b.calc_body_length()
            max_body_length = (body_length if body_length > max_body_length else max_body_length)
            if ID >= read.num_free_bodies:
                b.prescribed_kinematics = True
            bodies.append(b)
        bodies = np.array(bodies)

    #For debugging of rotations
    # q = Quaternion([1, 0, 0, 0])
    # q.random_orientation()
    # print(q.getAsVector())
    # print(2*q.gradToTourqueMatrix())



    # Set some more variables
    num_bodies = bodies.size
    # Nblobs = sum([x.Nblobs for x in bodies])
    # max_angle_shift = max_translation / max_body_length
    start_time = time.time()
    accepted_moves = 0
    if MCMC:
        acceptance_ratio = 0.5
        current_state_energy = potential_pycuda_user_defined.compute_total_energy(bodies, sample_r_vectors,
                                                                            periodic_length = periodic_length,
                                                                            debye_length_wall = read.debye_length_wall,
                                                                            repulsion_strength_wall = read.repulsion_strength_wall,
                                                                            debye_length = read.debye_length,
                                                                            repulsion_strength = read.repulsion_strength,
                                                                            weight = weight,
                                                                            blob_radius = blob_radius)

        # for each step in the Markov chain, disturb each body's location and orientation and obtain the new list of r_vectors
        # of each blob. Calculate the potential of the new state, and accept or reject it according to the Markov chain rules:
        # 1. if Ej < Ei, always accept the state  2. if Ej < Ei, accept the state according to the probability determined by
        # exp(-(Ej-Ei)/kT). Then record data.
        # Important: record data also when staying in the same state (i.e. when a sample state is rejected)
    if LM:
        old_noise = np.random.normal(0.0, 1.0, 12)
    for step in range(read.initial_step, read.n_steps):
        x1 = bodies[0].location
        x2 = bodies[1].location
        q1 = bodies[0].orientation
        q2 = bodies[1].orientation

        grad= potential_pycuda_user_defined.getGradient(np.concatenate((x1,x2)),np.concatenate((q1.getAsVector(),q2.getAsVector())))
        #print(grad)
        #Grad containes the gradient with respect to x, q
        #noise = np.random.normal(0.0, 1.0, 12)
        #now, grad shoule be converted to be of size 12
        #velocity_and_omega = grad + np.sqrt(2.0*kT/dt)*noise
        #print(np.max(grad*dt+np.sqrt(2.0*kT*dt)))
        #print(np.max(grad*dt))
        gradT = grad[0:6]
        gradA = grad[6:]
        print(np.max(np.abs(grad[6:])))

        #print(gradq)

        if not LM:
            noise = np.random.normal(0.0, 1.0, 12)
        else:
            new_noise = np.random.normal(0.0, 1.0, 12)
            noise = (new_noise + old_noise)*0.5
            old_noise = new_noise
        res_trans = 0.57006
        res_rot  = 14.4347
        for i, body in enumerate(bodies): # disturb bodies

            if body.prescribed_kinematics is False:
                #Here we propose a new location and orientation, by doing langevin dynamics

                trans_vel = -res_trans*gradT[i*3:i*3+3]+np.sqrt(res_trans*2.0*kT/dt)*noise[i*6:i*6+3]
                body.location_new = body.location + dt*trans_vel #the right indices?
                Q = body.orientation.gradToTourqueMatrix() #is this step problematic?

                rot_vel = - res_rot*Q @ gradA[i*4:(i+1)*4]+np.sqrt(res_rot*2.0*kT/dt)*noise[i*6+3:i*6+6]
                quaternion_dt = Quaternion.from_rotation(rot_vel*dt)
                body.orientation_new = quaternion_dt*body.orientation
                #print(body.location_new)
                #print(body.orientation_new)
            else:
                print("I am where I am not supposed to be")
                body.location_new = body.location
                body.orientation_new = body.orientation

        if MCMC:
            sample_state_energy = potential_pycuda_user_defined.compute_total_energy(bodies,
                                                                              sample_r_vectors,
                                                                              periodic_length = periodic_length,
                                                                              debye_length_wall = read.debye_length_wall,
                                                                              repulsion_strength_wall = read.repulsion_strength_wall,
                                                                              debye_length = read.debye_length,
                                                                              repulsion_strength = read.repulsion_strength,
                                                                              weight = weight,
                                                                              blob_radius = blob_radius)

            #print(np.exp(-(sample_state_energy - current_state_energy)))
            # accept or reject the sample state and collect data accordingly
            #print(acceptance_ratio)


            #print(np.exp(-(sample_state_energy - current_state_energy)))
            #Check here for overlaps in both new and old state
            dold = ph.shortestDist(x1,x2,q1,q2,L,R)

            x1 = bodies[0].location_new
            x2 = bodies[1].location_new
            q1 = bodies[0].orientation_new
            q2 = bodies[1].orientation_new
            dnew = ph.shortestDist(x1,x2,q1,q2,L,R)

            if (dold < 0) or (dnew < 0):
                print("Overlap occurs")
                print(dnew)
                print(dold)
                print("energies")
                print(sample_state_energy)
                print(current_state_energy)
                print(np.exp(-(sample_state_energy - current_state_energy)))
                raise Exception("Overlap occuring")



            if np.random.uniform(0.0, 1.0) < np.exp(-(sample_state_energy - current_state_energy) / kT):
                current_state_energy = sample_state_energy
                accepted_moves += 1
                acceptance_ratio = acceptance_ratio * 0.95 + 0.05
                for body in bodies:
                    body.location, body.orientation = body.location_new, body.orientation_new
            else:
                acceptance_ratio = acceptance_ratio * 0.95

            # Scale max_translation
            if step < 0 and step < read.initial_step // 2 and acceptance_ratio > 0.5:
                max_translation = max_translation * 1.02
                max_angle_shift = max_translation / max_body_length
            elif step < 0 and step < read.initial_step // 2:
                max_translation = max_translation * 0.98
                max_angle_shift = max_translation / max_body_length
        else:
            for body in bodies:
              body.location, body.orientation = body.location_new, body.orientation_new
              accepted_moves +=1

        # Save data if...
        if (step % read.n_save) == 0 and step >= 0:
            elapsed_time = time.time() - start_time
            print('MCMC, step = ', step, ', wallclock time = ', time.time() - start_time, ', acceptance ratio = ', accepted_moves / (step+1.0-read.initial_step))
            # For each type of structure save locations and orientations to one file
            body_offset = 0
            if read.save_clones == 'one_file_per_step':
                for i, ID in enumerate(read.structures_ID):
                    name = read.output_name + '.' + ID + '.' + str(step).zfill(8) + '.clones'
                    with open(name, 'w') as f_ID:
                        f_ID.write(str(body_types[i]) + '\n')
                        for j in range(body_types[i]):
                            orientation = bodies[body_offset + j].orientation.entries
                            f_ID.write('%s %s %s %s %s %s %s\n' % (bodies[body_offset + j].location[0],
                                                     bodies[body_offset + j].location[1],
                                                     bodies[body_offset + j].location[2],
                                                     orientation[0],
                                                     orientation[1],
                                                     orientation[2],
                                                     orientation[3]))
                        body_offset += body_types[i]
            elif read.save_clones == 'one_file':
                for i, ID in enumerate(read.structures_ID):
                    name = read.output_name + '.' + ID + '.config'
                    if step == 0:
                        status = 'w'
                    else:
                        status = 'a'
                    with open(name, status) as f_ID:
                        f_ID.write(str(body_types[i]) + '\n')
                        for j in range(body_types[i]):
                            orientation = bodies[body_offset + j].orientation.entries
                            f_ID.write('%s %s %s %s %s %s %s\n' % (bodies[body_offset + j].location[0],
                                                                 bodies[body_offset + j].location[1],
                                                                 bodies[body_offset + j].location[2],
                                                                 orientation[0],
                                                                 orientation[1],
                                                                 orientation[2],
                                                                 orientation[3]))
                        body_offset += body_types[i]
         # else:
         #      print('Error, save_clones =', read.save_clones, 'is not implemented.')
         #      print('Use \"one_file_per_step\" or \"one_file\". \n')
         #      break

    # Save final data if...
    if ((step+1) % read.n_save) == 0 and step >= 0:
        print('MCMC, step = ', step+1, ', wallclock time = ', time.time() - start_time, ', acceptance ratio = ', accepted_moves / (step+2.0-read.initial_step))
         # For each type of structure save locations and orientations to one file
        body_offset = 0
        if read.save_clones == 'one_file_per_step':
             for i, ID in enumerate(read.structures_ID):
                 name = read.output_name + '.' + ID + '.' + str(step+1).zfill(8) + '.clones'
                 with open(name, 'w') as f_ID:
                     f_ID.write(str(body_types[i]) + '\n')
                     for j in range(body_types[i]):
                         orientation = bodies[body_offset + j].orientation.entries
                         f_ID.write('%s %s %s %s %s %s %s\n' % (bodies[body_offset + j].location[0],
                                                           bodies[body_offset + j].location[1],
                                                           bodies[body_offset + j].location[2],
                                                           orientation[0],
                                                           orientation[1],
                                                           orientation[2],
                                                           orientation[3]))
                     body_offset += body_types[i]

        elif read.save_clones == 'one_file':
            for i, ID in enumerate(read.structures_ID):
                 name = read.output_name + '.' + ID + '.config'
                 if step+1 == 0:
                     status = 'w'
                 else:
                     status = 'a'
                 with open(name, status) as f_ID:
                     f_ID.write(str(body_types[i]) + '\n')
                     for j in range(body_types[i]):
                         orientation = bodies[body_offset + j].orientation.entries
                         f_ID.write('%s %s %s %s %s %s %s\n' % (bodies[body_offset + j].location[0],
                                                       bodies[body_offset + j].location[1],
                                                       bodies[body_offset + j].location[2],
                                                       orientation[0],
                                                       orientation[1],
                                                       orientation[2],
                                                       orientation[3]))
                     body_offset += body_types[i]
        else:
            print('Error, save_clones =', read.save_clones, 'is not implemented.')
            print('Use \"one_file_per_step\" or \"one_file\". \n')

    end_time = time.time() - start_time
    print('\nacceptance ratio = ', accepted_moves / (step+2.0-read.initial_step))
    print('accepted_moves = ', accepted_moves)
    print('Total time = ', end_time)

    # Save wallclock time
    with open(read.output_name + '.time', 'w') as f:
        f.write(str(time.time() - start_time) + '\n')
    # Save acceptance ratio
    with open(read.output_name + '.MCMC_info', 'w') as f:
        f.write('acceptance ratio = ' + str(accepted_moves / (step+2.0-read.initial_step)) + '\n')
        f.write('accepted_moves = ' +  str(accepted_moves) + '\n')
        #f.write('final max_translation = ' +  str(max_translation) + '\n')
        #f.write('final max_angle_shift = ' +  str(max_angle_shift) + '\n')
