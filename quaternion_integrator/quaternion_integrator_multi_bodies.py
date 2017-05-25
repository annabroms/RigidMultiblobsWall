'''
Integrator for several rigid bodies.
'''
import numpy as np
import math as m
import scipy.sparse.linalg as spla
from functools import partial

from quaternion import Quaternion
from stochastic_forcing import stochastic_forcing as stochastic
from mobility import mobility as mob

import scipy

class QuaternionIntegrator(object):
  '''
  Integrator that timesteps using deterministic forwars Euler scheme.
  '''  
  def __init__(self, bodies, Nblobs, scheme, tolerance = None): 
    ''' 
    Init object 
    '''
    self.bodies = bodies
    self.Nblobs = Nblobs
    self.scheme = scheme
    self.mobility_bodies = np.empty((len(bodies), 6, 6))

    # Other variables
    self.get_blobs_r_vectors = None
    self.mobility_blobs = None
    self.force_torque_calculator = None
    self.calc_K_matrix_bodies = None
    self.linear_operator = None
    self.eta = None
    self.a = None
    self.velocities = None
    self.velocities_previous_step = None
    self.first_step = True
    self.kT = 0.0
    self.tolerance = 1e-08
    self.rf_delta = 1e-05
    self.invalid_configuration_count = 0

    # Optional variables
    self.build_stochastic_block_diagonal_preconditioner = None
    self.periodic_length = None
    self.calc_slip = None
    self.calc_force_torque = None
    self.mobility_inv_blobs = None
    self.first_guess = None
    self.preconditioner = None
    self.mobility_vector_prod = None    
    if tolerance is not None:
      self.tolerance = tolerance
      self.rf_delta = 0.1 * np.power(self.tolerance, 1.0/3.0)
    return 

  def advance_time_step(self, dt):
    '''
    Advance time step with integrator self.scheme
    '''
    return getattr(self, self.scheme)(dt)
    

  def deterministic_forward_euler(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic forward Euler scheme. 
    The function uses gmres to solve the rigid body equations.

    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Call preprocess
      preprocess_result = self.preprocess(self.bodies)

      # Solve mobility problem
      sol_precond = self.solve_mobility_problem(x0 = self.first_guess, save_first_guess = True)
      
      # Extract velocities
      velocities = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      # Update location orientation 
      for k, b in enumerate(self.bodies):
        b.location_new = b.location + velocities[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation
        
      # Call postprocess
      postprocess_result = self.postprocess(self.bodies)

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        for b in self.bodies:
          b.location = b.location_new
          b.orientation = b.orientation_new
        return
      
      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return
      

  def deterministic_forward_euler_dense_algebra(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic forward Euler scheme. 
    The function uses dense algebra methods to solve the equations.
    
    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Call preprocess
      preprocess_result = self.preprocess(self.bodies)

      # Solve mobility problem
      velocities, mobility_bodies = self.solve_mobility_problem_dense_algebra()

      # Update location orientation 
      for k, b in enumerate(self.bodies):
        b.location_new = b.location + velocities[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation
        
      # Call postprocess
      postprocess_result = self.postprocess(self.bodies)

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        for b in self.bodies:
          b.location = b.location_new
          b.orientation = b.orientation_new
        return

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return
      
  
  def deterministic_adams_bashforth(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic Adams-Bashforth of
    order two scheme. The function uses gmres to solve the rigid body equations.

    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Call preprocess
      preprocess_result = self.preprocess(self.bodies)

      # Solve mobility problem
      sol_precond = self.solve_mobility_problem(x0 = self.first_guess, save_first_guess = True)

      # Extract velocities
      velocities = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      # Update location and orientation
      if self.first_step == False:
        # Use Adams-Bashforth
        for k, b in enumerate(self.bodies):
          b.location_new = b.location + (1.5 * velocities[6*k:6*k+3] - 0.5 * self.velocities_previous_step[6*k:6*k+3]) * dt
          quaternion_dt = Quaternion.from_rotation((1.5 * velocities[6*k+3:6*k+6] \
                                                      - 0.5 * self.velocities_previous_step[6*k+3:6*k+6]) * dt)
          b.orientation_new = quaternion_dt * b.orientation
      else:
        # Use forward Euler
        for k, b in enumerate(self.bodies):
          b.location_new = b.location + velocities[6*k:6*k+3] * dt
          quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
          b.orientation_new = quaternion_dt * b.orientation              

      # Call postprocess
      postprocess_result = self.postprocess(self.bodies)

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        # Save velocities for next step
        self.first_step = False
        self.velocities_previous_step = velocities
        for b in self.bodies:
          b.location = b.location_new
          b.orientation = b.orientation_new          
        return
    
      self.invalid_configuration_count += 1
      print 'Invalid configuration'      
    return


  def stochastic_first_order_RFD(self, dt): 
    ''' 
    Take a time step of length dt using a stochastic
    first order Randon Finite Difference (RFD) scheme.

    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Call preprocess
      preprocess_result = self.preprocess(self.bodies)
      
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
        b.orientation_old = b.orientation

      # Generate random vector
      rfd_noise = np.random.normal(0.0, 1.0, len(self.bodies) * 6)     

      # Get blobs vectors
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      # Build preconditioners
      PC_partial, mobility_pc_partial, P_inv_mult = self.build_block_diagonal_preconditioners_det_stoch(self.bodies, 
                                                                                                        r_vectors_blobs, 
                                                                                                        self.Nblobs, 
                                                                                                        self.eta, 
                                                                                                        self.a,
                                                                                                        periodic_length=self.periodic_length)

      # Add noise contribution sqrt(2kT/dt)*N^{1/2}*W
      velocities_noise, it_lanczos = stochastic.stochastic_forcing_lanczos(factor = np.sqrt(2*self.kT / dt),
                                                                           tolerance = self.tolerance, 
                                                                           dim = self.Nblobs * 3, 
                                                                           mobility_mult = mobility_pc_partial,
                                                                           L_mult = P_inv_mult)

      # Solve mobility problem
      sol_precond = self.solve_mobility_problem(noise = velocities_noise, x0 = self.first_guess, save_first_guess = True, PC_partial = PC_partial)

      # Extract velocities
      velocities = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      # Update configuration for rfd 
      force_rfd = np.copy(rfd_noise) 
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + rfd_noise[k*6 : k*6+3] * (-self.rf_delta * 0.5 * b.body_length)
        quaternion_dt = Quaternion.from_rotation(rfd_noise[(k*6+3):(k*6+6)] * (-self.rf_delta * 0.5))
        b.orientation = quaternion_dt * b.orientation_old
        force_rfd[k*6 : k*6+3] /= b.body_length
        

      # Add thermal drift contribution with N at x = x - random_displacement
      System_size = self.Nblobs * 3 + len(self.bodies) * 6
      sol_precond = self.solve_mobility_problem(RHS = np.reshape(np.concatenate([np.zeros(3*self.Nblobs), -force_rfd]), (System_size)), PC_partial = PC_partial)

      # Update configuration for rfd 
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + rfd_noise[k*6 : k*6+3] * (self.rf_delta * 0.5 * b.body_length)
        quaternion_dt = Quaternion.from_rotation(rfd_noise[(k*6+3):(k*6+6)] * (self.rf_delta * 0.5))
        b.orientation = quaternion_dt * b.orientation_old

      # Modify RHS for drift solve
      # Set linear operators 
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)
      linear_operator_partial = partial(self.linear_operator, 
                                        bodies=self.bodies, 
                                        r_vectors=r_vectors_blobs, 
                                        eta=self.eta, 
                                        a=self.a, 
                                        periodic_length=self.periodic_length)
      A = spla.LinearOperator((System_size, System_size), matvec = linear_operator_partial, dtype='float64')
      RHS = np.reshape(np.concatenate([np.zeros(3*self.Nblobs), -force_rfd]), (System_size)) - A * sol_precond

      # Add thermal drift contribution with N at x = x + random_displacement
      sol_precond = self.solve_mobility_problem(RHS = RHS, PC_partial = PC_partial)

      # Extract velocities
      velocities_drift = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      # Add all velocity contributions
      velocities += (self.kT / self.rf_delta) * velocities_drift
      
      # Update location orientation 
      for k, b in enumerate(self.bodies):
        b.location_new = b.location_old + velocities[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation_old

      # Call postprocess
      postprocess_result = self.postprocess(self.bodies)

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        for b in self.bodies:
          b.location = b.location_new
          b.orientation = b.orientation_new
        return
      else:
        for b in self.bodies:
          b.location = b.location_old
          b.orientation = b.orientation_old

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return


  def stochastic_adams_bashforth(self, dt): 
    ''' 
    Take a time step of length dt using a stochastic
    first order Randon Finite Difference (RFD) scheme.

    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Call preprocess
      preprocess_result = self.preprocess(self.bodies)
      
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
        b.orientation_old = b.orientation

      # Generate random vector
      rfd_noise = np.random.normal(0.0, 1.0, len(self.bodies) * 6)     

      # Get blobs vectors
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      # Build preconditioners
      PC_partial, mobility_pc_partial, P_inv_mult = self.build_block_diagonal_preconditioners_det_stoch(self.bodies, 
                                                                                                        r_vectors_blobs, 
                                                                                                        self.Nblobs, 
                                                                                                        self.eta, 
                                                                                                        self.a,
                                                                                                        periodic_length=self.periodic_length)

      # Add noise contribution sqrt(2kT/dt)*N^{1/2} * W
      velocities_noise, it_lanczos = stochastic.stochastic_forcing_lanczos(factor = np.sqrt(2*self.kT / dt),
                                                                           tolerance = self.tolerance, 
                                                                           dim = self.Nblobs * 3, 
                                                                           mobility_mult = mobility_pc_partial,
                                                                           L_mult = P_inv_mult)

      # Solve stochastic mobility problem
      System_size = self.Nblobs * 3 + len(self.bodies) * 6
      sol_precond = self.solve_mobility_problem(RHS = np.zeros(System_size), noise = velocities_noise, PC_partial = PC_partial)

      # Extract stochastic velocities
      velocities_stoch = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      # Solve deterministic mobility problem
      sol_precond = self.solve_mobility_problem(x0 = self.first_guess, save_first_guess = True, PC_partial = PC_partial)

      # Extract deterministic velocities
      velocities_det = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      # Update configuration for rfd 
      force_rfd = np.copy(rfd_noise) 
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + rfd_noise[k*6 : k*6+3] * (-self.rf_delta * 0.5 * b.body_length)
        quaternion_dt = Quaternion.from_rotation(rfd_noise[(k*6+3):(k*6+6)] * (-self.rf_delta * 0.5))
        b.orientation = quaternion_dt * b.orientation_old
        force_rfd[k*6 : k*6+3] /= b.body_length

      # Add thermal drift contribution with N at x = x - random_displacement
      sol_precond = self.solve_mobility_problem(RHS = np.reshape(np.concatenate([np.zeros(3*self.Nblobs), -force_rfd]), (System_size)), PC_partial = PC_partial)

      # Update configuration for rfd 
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + rfd_noise[k*6 : k*6+3] * (self.rf_delta * 0.5 * b.body_length)
        quaternion_dt = Quaternion.from_rotation(rfd_noise[(k*6+3):(k*6+6)] * (self.rf_delta * 0.5))
        b.orientation = quaternion_dt * b.orientation_old

      # Modify RHS for drift solve
      # Set linear operators 
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)
      linear_operator_partial = partial(self.linear_operator, 
                                        bodies=self.bodies, 
                                        r_vectors=r_vectors_blobs, 
                                        eta=self.eta, 
                                        a=self.a, 
                                        periodic_length=self.periodic_length)
      A = spla.LinearOperator((System_size, System_size), matvec = linear_operator_partial, dtype='float64')
      RHS = np.reshape(np.concatenate([np.zeros(3*self.Nblobs), -force_rfd]), (System_size)) - A * sol_precond

      # Add thermal drift contribution with N at x = x + random_displacement
      sol_precond = self.solve_mobility_problem(RHS = RHS, PC_partial = PC_partial)

      # Extract velocities
      velocities_drift = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      # Add all velocity contributions
      velocities_stoch += (self.kT / self.rf_delta) * velocities_drift
            
      # Update location and orientation
      if self.first_step == False:
        # Use Adams-Bashforth
        for k, b in enumerate(self.bodies):
          b.location_new = b.location_old + (1.5 * velocities_det[6*k:6*k+3] - 0.5 * self.velocities_previous_step[6*k:6*k+3] + velocities_stoch[6*k:6*k+3]) * dt
          quaternion_dt = Quaternion.from_rotation((1.5 * velocities_det[6*k+3:6*k+6] - 0.5 * self.velocities_previous_step[6*k+3:6*k+6] + velocities_stoch[6*k+3:6*k+6]) * dt)
          b.orientation_new = quaternion_dt * b.orientation_old
      else:
        # Use forward Euler
        for k, b in enumerate(self.bodies):
          b.location_new = b.location_old + (velocities_det[6*k:6*k+3] + velocities_stoch[6*k:6*k+3]) * dt
          quaternion_dt = Quaternion.from_rotation((velocities_det[6*k+3:6*k+6] + velocities_stoch[6*k+3:6*k+6]) * dt)
          b.orientation_new = quaternion_dt * b.orientation_old

      # Call postprocess
      postprocess_result = self.postprocess(self.bodies)

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        self.first_step = False
        self.velocities_previous_step = velocities_det
        for b in self.bodies:
          b.location = b.location_new
          b.orientation = b.orientation_new
        return
      else:
        for b in self.bodies:
          b.location = b.location_old
          b.orientation = b.orientation_old

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return


  def stochastic_first_order_RFD_dense_algebra(self, dt): 
    ''' 
    Take a time step of length dt using a stochastic
    first order Randon Finite Difference (RFD) scheme.
    The function uses dense algebra methods to solve the equations.

    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Call preprocess
      preprocessor_result = self.preprocess(self.bodies)

      # Solve mobility problem
      velocities, mobility_bodies = self.solve_mobility_problem_dense_algebra()

      # Generate random vector
      rfd_noise = np.random.normal(0.0, 1.0, len(self.bodies) * 6)     

      # Add noise contribution sqrt(2kT/dt)*N^{1/2}*W
      velocities += stochastic.stochastic_forcing_eig(mobility_bodies, factor = np.sqrt(2*self.kT / dt))

      # Update configuration for rfd
      force_rfd = np.copy(rfd_noise) 
      for k, b in enumerate(self.bodies):
        b.location_new = b.location + rfd_noise[k*6 : k*6+3] * (self.rf_delta * b.body_length)
        quaternion_dt = Quaternion.from_rotation(rfd_noise[(k*6+3):(k*6+6)] * (self.rf_delta))
        b.orientation_new = quaternion_dt * b.orientation
        force_rfd[k*6 : k*6+3] /= b.body_length

      # Compute bodies' mobility at new configuration
      # Get blobs coordinates
      r_vectors_blobs = np.empty((self.Nblobs, 3))
      offset = 0
      for b in self.bodies:
        r_vectors_blobs[offset:(offset+b.Nblobs)] = b.get_r_vectors(b.location_new, b.orientation_new)
        offset += b.Nblobs

      # Calculate mobility (M) at the blob level
      mobility_blobs = self.mobility_blobs(r_vectors_blobs, self.eta, self.a)

      # Calculate resistance at the blob level (use np.linalg.inv or np.linalg.pinv)
      resistance_blobs = np.linalg.inv(mobility_blobs)

      # Calculate block-diagonal matrix K
      K = np.zeros((3*self.Nblobs, 6*len(self.bodies)))
      offset = 0
      for k, b in enumerate(self.bodies):
        K[3*offset:3*(offset+b.Nblobs), 6*k:6*k+6] = b.calc_K_matrix(location = b.location_new, orientation = b.orientation_new)
        offset += b.Nblobs
     
      # Calculate mobility (N) at the body level. Use np.linalg.inv or np.linalg.pinv
      mobility_bodies_new = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)), rcond=1e-14)

      # Add thermal drift to velocity
      velocities += (self.kT / self.rf_delta) * np.dot(mobility_bodies_new - mobility_bodies, force_rfd) 
      
      # Update location orientation 
      for k, b in enumerate(self.bodies):
        b.location_new = b.location + velocities[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation
        
      # Call postprocess
      postprocess_result = self.postprocess(self.bodies)

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        for b in self.bodies:
          b.location = b.location_new
          b.orientation = b.orientation_new
        return

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return


  def stochastic_traction_EM(self, dt):
    ''' 
    Take a time step of length dt using a stochastic first order 
    Randon Finite Difference (RFD) scheme. This function uses
    a traction method to compute the RFD. 
    
    The computational cost is 2 rigid solves + 1 lanczos call 
    + 2 blobs mobility product + 4 products with the geometric matrix K.

    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    '''
    while True:
      # Call preprocess
      preprocess_result = self.preprocess(self.bodies)

      rfd_noise = np.random.normal(0.0, 1.0, 6*len(self.bodies))
      W = np.empty_like(rfd_noise)

      # Save initial configuration and scale noise
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
        b.orientation_old = b.orientation
        W[k*6 : k*6+3] = rfd_noise[k*6 : k*6+3] * (self.kT / b.body_length)
        W[(k*6+3):(k*6+6)] = rfd_noise[(k*6+3):(k*6+6)] * self.kT

      # Set RHS for RFD increments   
      System_size = self.Nblobs * 3 + len(self.bodies) * 6
      RAND_RHS = np.zeros(System_size)
      RAND_RHS[3*self.Nblobs:System_size] = -1.0*W

      # Get blobs vectors
      r_vectors_blobs_n = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      # Build preconditioners
      PC_partial, mobility_pc_partial, P_inv_mult = self.build_block_diagonal_preconditioners_det_stoch(self.bodies, 
                                                                                                        r_vectors_blobs_n, 
                                                                                                        self.Nblobs, 
                                                                                                        self.eta, 
                                                                                                        self.a,
                                                                                                        periodic_length=self.periodic_length)

      # Generate RFD increments
      sol_precond = self.solve_mobility_problem(RHS = RAND_RHS, PC_partial = PC_partial)
      U_RFD = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))
      Lam_RFD = sol_precond[0:3*self.Nblobs]

      # compute M*Lambda_rfd
      MxLam = self.mobility_vector_prod(r_vectors_blobs_n, Lam_RFD, self.eta, self.a, periodic_length = self.periodic_length)
      # compute K^T*Lambda_rfd
      KTxLam = self.K_matrix_T_vector_prod(self.bodies, Lam_RFD, self.Nblobs)
      # compute K*U_rfd
      KxU = self.K_matrix_vector_prod(self.bodies, U_RFD, self.Nblobs)

      # Compute RFD bits
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + rfd_noise[6*k:6*k+3] * (self.rf_delta * b.body_length)
        quaternion_dt = Quaternion.from_rotation(rfd_noise[6*k+3:6*k+6] * self.rf_delta)
        b.orientation = quaternion_dt * b.orientation_old
      r_vectors_blobs_RFD = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      # compute (M_rfd-M)*Lambda_rfd
      DxM = self.mobility_vector_prod(r_vectors_blobs_RFD, Lam_RFD, self.eta, self.a, periodic_length = self.periodic_length) - MxLam
      # compute (K_rfd^T - K^T)*Lambda_rfd
      DxKT = self.K_matrix_T_vector_prod(self.bodies, Lam_RFD, self.Nblobs) - KTxLam
      # compute (K_rfd - K)*U_rfd
      DxK = np.reshape( self.K_matrix_vector_prod(self.bodies, U_RFD, self.Nblobs) - KxU, 3*self.Nblobs)

      # reset locs and thetas
      for k, b in enumerate(self.bodies):
        b.location = b.location_old
        b.orientation = b.orientation_old

      # Add noise contribution sqrt(2kT/dt)*N^{1/2}*W
      slip_noise, it_lanczos = stochastic.stochastic_forcing_lanczos(factor = np.sqrt(2.0*self.kT / dt),
                                                                     tolerance = self.tolerance,
                                                                     dim = self.Nblobs * 3,
                                                                     mobility_mult = mobility_pc_partial,
                                                                     L_mult = P_inv_mult)

      rand_slip = slip_noise + (1.0 / self.rf_delta) * (DxM - DxK)
      rand_force = (-1.0 / self.rf_delta) * DxKT

      # Solve mobility problem with drift
      sol_precond_new = self.solve_mobility_problem(noise = rand_slip, 
                                                    noise_FT = rand_force, 
                                                    x0 = self.first_guess, 
                                                    save_first_guess = True, 
                                                    PC_partial = PC_partial)
      velocities_new = np.reshape(sol_precond_new[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      # Update location and orientation
      for k, b in enumerate(self.bodies):
        b.location_new = b.location_old + velocities_new[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities_new[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation_old

      # Call postprocess
      postprocess_result = self.postprocess(self.bodies)

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        for k, b in enumerate(self.bodies):
          b.location = b.location_new
          b.orientation = b.orientation_new
        return
      else:
        for b in self.bodies:
          b.location = b.location_old
          b.orientation = b.orientation_old

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return

  def Fixman(self, dt):
    ''' 
    Take a time step of length dt using a stochastic
    first order Randon Finite Difference (RFD) schame.
    This scheme uses dense algebra methods.

    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    '''
    while True:
      # Call preprocess
      preprocess_result = self.preprocess(self.bodies)

      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
        b.orientation_old = b.orientation
     
      # Solve mobility problem
      velocities_mid, mobility_bodies = self.solve_mobility_problem_dense_algebra()

      # Generate random vector
      W1 = np.random.normal(0.0, 1.0, len(self.bodies) * 6)
      W_cor = W1 + np.random.normal(0.0, 1.0, len(self.bodies) * 6);

      # Compute noise contribution for pred. step sqrt(2kT/dt)*N^{1/2}*W1
      Nhalf_W1 = stochastic.stochastic_forcing_eig(mobility_bodies, factor = np.sqrt(4*self.kT / dt),z = W1)
     
      # Compute noise contribution for cor. step sqrt(2kT/dt)*N^{1/2}*W1
      Nhalf_Wcor = stochastic.stochastic_forcing_eig(mobility_bodies, factor = np.sqrt(self.kT / dt),z = W_cor)
      Ninvhalf_cor = np.dot(np.linalg.pinv(mobility_bodies, rcond=1e-14),Nhalf_Wcor)
     
      velocities_mid += Nhalf_W1

      # Update location orientation to mid point
      for k, b in enumerate(self.bodies):
	b.location = b.location_old + velocities_mid[6*k:6*k+3] * dt * 0.5
	quaternion_dt = Quaternion.from_rotation((velocities_mid[6*k+3:6*k+6]) * dt * 0.5)
	b.orientation = quaternion_dt * b.orientation_old
   
      # Solve mobility problem predictor step
      velocities_new, mobility_bodies = self.solve_mobility_problem_dense_algebra()
      velocities_new += np.dot(mobility_bodies,Ninvhalf_cor)
   
      # Update location orientation to end point
      for k, b in enumerate(self.bodies):
	b.location_new = b.location_old + velocities_new[6*k:6*k+3] * dt
	quaternion_dt = Quaternion.from_rotation((velocities_new[6*k+3:6*k+6]) * dt)
	b.orientation_new = quaternion_dt * b.orientation_old

      # Call postprocess
      postprocess_result = self.postprocess(self.bodies)

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        self.first_step = False
        self.velocities_previous_step = velocities_new
        for k, b in enumerate(self.bodies):
          b.location = b.location_new
          b.orientation = b.orientation_new
        return
      else:
        for b in self.bodies:
          b.location = b.location_old
          b.orientation = b.orientation_old

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return


  def stochastic_traction_AB(self, dt): 
    ''' 
    Take a time step of length dt using a stochastic
    Adams-Bashfoth scheme and using a traction method
    to compute the RFD. 

    The computational cost of this method is 3 rigid solves
    + 1 lanczos call + 2 blobs mobility product + 4 products with the geometric matrix K.

    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Call preprocess
      preprocess_result = self.preprocess(self.bodies)
      
      rfd_noise = np.random.normal(0.0, 1.0, 6*len(self.bodies))
      W = np.empty_like(rfd_noise)

      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
        b.orientation_old = b.orientation
        W[k*6 : k*6+3] = rfd_noise[k*6 : k*6+3] * (self.kT / b.body_length)
	W[(k*6+3):(k*6+6)] = rfd_noise[(k*6+3):(k*6+6)] * self.kT   
        
      # Set RHS for RFD increments   
      System_size = self.Nblobs * 3 + len(self.bodies) * 6
      RAND_RHS = np.zeros(System_size)
      RAND_RHS[3*self.Nblobs:System_size] = -1.0 * W
      
      # Get blobs vectors
      r_vectors_blobs_n = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      # Build preconditioners
      PC_partial, mobility_pc_partial, P_inv_mult = self.build_block_diagonal_preconditioners_det_stoch(self.bodies, 
                                                                                                        r_vectors_blobs_n, 
                                                                                                        self.Nblobs, 
                                                                                                        self.eta, 
                                                                                                        self.a,
                                                                                                        periodic_length=self.periodic_length)

      # Generate RFD increments
      sol_precond = self.solve_mobility_problem(RHS = RAND_RHS, PC_partial = PC_partial)
      U_RFD = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))
      Lam_RFD = sol_precond[0:3*self.Nblobs]
         
      # compute M*Lambda_rfd
      MxLam = self.mobility_vector_prod(r_vectors_blobs_n, Lam_RFD, self.eta, self.a, periodic_length = self.periodic_length)
      # compute K^T*Lambda_rfd
      KTxLam = self.K_matrix_T_vector_prod(self.bodies, Lam_RFD, self.Nblobs)
      # compute K*U_rfd
      KxU = self.K_matrix_vector_prod(self.bodies, U_RFD, self.Nblobs)
      
      # Compute RFD bits
      for k, b in enumerate(self.bodies):
	b.location = b.location_old + rfd_noise[6*k:6*k+3] * (self.rf_delta * b.body_length)
	quaternion_dt = Quaternion.from_rotation(rfd_noise[6*k+3:6*k+6] * self.rf_delta)
        b.orientation = quaternion_dt * b.orientation_old    
	
      # compute (M_rfd-M)*Lambda_rfd
      r_vectors_blobs_RFD = self.get_blobs_r_vectors(self.bodies, self.Nblobs)
      DxM = self.mobility_vector_prod(r_vectors_blobs_RFD, Lam_RFD, self.eta, self.a, periodic_length = self.periodic_length) - MxLam
      # compute (K_rfd^T - K^T)*Lambda_rfd
      DxKT = self.K_matrix_T_vector_prod(self.bodies, Lam_RFD, self.Nblobs) - KTxLam
      # compute (K_rfd - K)*U_rfd
      DxK = np.reshape( self.K_matrix_vector_prod(self.bodies, U_RFD, self.Nblobs) - KxU, 3*self.Nblobs)
          
      # reset locs and thetas
      for k, b in enumerate(self.bodies):
	b.location = b.location_old
	b.orientation = b.orientation_old

      # Add noise contribution sqrt(2kT/dt)*N^{1/2}*W
      slip_noise, it_lanczos = stochastic.stochastic_forcing_lanczos(factor = np.sqrt(2.0*self.kT / dt),
                                                                     tolerance = self.tolerance, 
                                                                     dim = self.Nblobs * 3, 
                                                                     mobility_mult = mobility_pc_partial,
                                                                     L_mult = P_inv_mult)
      
      rand_slip = (1.0 / self.rf_delta)* (DxM - DxK)
      rand_force = (-1.0 / self.rf_delta)* DxKT


      # Solve mobility problem with drift
      sol_precond_new = self.solve_mobility_problem(noise = rand_slip, 
                                                    noise_FT = rand_force, 
                                                    x0 = self.first_guess, 
                                                    save_first_guess = True,
                                                    PC_partial = PC_partial)
      velocities_new = np.reshape(sol_precond_new[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))
      sol_precond_rand = self.solve_mobility_problem(RHS = np.concatenate([-1.0*slip_noise, np.zeros(len(self.bodies) * 6)]), PC_partial = PC_partial)
      velocities_noise = np.reshape(sol_precond_rand[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))
      
      if self.first_step == False:
        # Use Adams-Bashforth
        velocities_AB = 1.5*velocities_new + velocities_noise - 0.5 * self.velocities_previous_step
      else: 
        # Use EM
        velocities_AB = velocities_new + velocities_noise
            
      # Update location and orientation
      for k, b in enumerate(self.bodies):
	b.location_new = b.location_old + velocities_AB[6*k:6*k+3] * dt
	quaternion_dt = Quaternion.from_rotation((velocities_AB[6*k+3:6*k+6]) * dt)
	b.orientation_new = quaternion_dt * b.orientation_old
      
      # Call postprocess
      postprocess_result = self.postprocess(self.bodies)

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        self.first_step = False
        self.velocities_previous_step = velocities_new
        for k, b in enumerate(self.bodies):
          b.location = b.location_new
          b.orientation = b.orientation_new
        return
      else:
        for b in self.bodies:
          b.location = b.location_old
          b.orientation = b.orientation_old

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return


  def stochastic_Slip_Trapz(self, dt):
    ''' 
    Take a time step of length dt using a stochastic 
    trapezoidal method. The thermal drift is handle
    with a slip method.

    The computational cost of this scheme is 3 rigid solves
    + 1 lanczos call + 2 blob mobility product + 2 products with the geometri matrix K.

    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    '''
    while True:
      # Call preprocess
      preprocess_result = self.preprocess(self.bodies)

      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
        b.orientation_old = b.orientation

      # Generate random vector
      W1 = np.random.normal(0.0, 1.0, self.Nblobs*3)
      W_slip = np.random.normal(0.0, 1.0, self.Nblobs*3)

      # Compute M at time level n
      r_vectors_blobs_n = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      #compute M*W to be used by the corrector step
      MxW_slip = self.mobility_vector_prod(r_vectors_blobs_n, W_slip, self.eta, self.a, periodic_length = self.periodic_length)
      #compute K^T*W to be used by the corrector step
      KTxW_slip = self.K_matrix_T_vector_prod(self.bodies,W_slip, self.Nblobs)

      # Build preconditioners
      PC_partial, mobility_pc_partial, P_inv_mult = self.build_block_diagonal_preconditioners_det_stoch(self.bodies, 
                                                                                                        r_vectors_blobs_n, 
                                                                                                        self.Nblobs, 
                                                                                                        self.eta, 
                                                                                                        self.a,
                                                                                                        periodic_length=self.periodic_length)

      # Calc noise contributions M^{1/2}*W1 and M^{1/2}*(W1+W3)
      velocities_noise_W1, it_lanczos = stochastic.stochastic_forcing_lanczos(factor = np.sqrt(2*self.kT / dt),
                                                                           tolerance = self.tolerance,
                                                                           dim = self.Nblobs * 3,
                                                                           mobility_mult = mobility_pc_partial,
                                                                           L_mult = P_inv_mult,
                                                                           z = W1)


      # Solve mobility problem
      sol_precond = self.solve_mobility_problem(noise = velocities_noise_W1, 
                                                x0 = self.first_guess, 
                                                save_first_guess = True,
                                                PC_partial = PC_partial)
      # Extract velocities
      velocities_1 = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      # Solve mobility problem
      slip_precond_rfd = self.solve_mobility_problem(RHS = np.concatenate([-1.0*W_slip, np.zeros(len(self.bodies) * 6)]), PC_partial = PC_partial)
      # Extract velocities
      W_RFD = np.reshape(slip_precond_rfd[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      # Update configuration for rfd 
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + W_RFD[k*6 : k*6+3] * self.rf_delta
        quaternion_dt = Quaternion.from_rotation(W_RFD[(k*6+3):(k*6+6)] * self.rf_delta )
        b.orientation = quaternion_dt * b.orientation_old

      # Compute M at RFD time level
      r_vectors_blobs_rfd = self.get_blobs_r_vectors(self.bodies, self.Nblobs)
      #compute M*W to be used by the corrector step
      M_rfdxW_slip = self.mobility_vector_prod(r_vectors_blobs_rfd, W_slip, self.eta, self.a, periodic_length = self.periodic_length)
      #compute K^T*W to be used by the corrector step
      KT_rfdxW_slip = self.K_matrix_T_vector_prod(self.bodies, W_slip, self.Nblobs)

      rand_slip_cor = velocities_noise_W1 + (2.0*self.kT / self.rf_delta) * (M_rfdxW_slip - MxW_slip)
      rand_force_cor = -2.0 * (self.kT / self.rf_delta) * (KT_rfdxW_slip - KTxW_slip)

      # Update location orientation to mid point
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + velocities_1[k*6 : k*6+3] * dt
        quaternion_dt = Quaternion.from_rotation(velocities_1[(k*6+3):(k*6+6)] * dt )
        b.orientation = quaternion_dt * b.orientation_old

      # Solve mobility problem at the corrector step
      sol_precond_cor = self.solve_mobility_problem(noise = rand_slip_cor, noise_FT = rand_force_cor, x0 = self.first_guess, save_first_guess = True)

      # Extract velocities
      velocities_2 = np.reshape(sol_precond_cor[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      velocities_new = 0.5 * (velocities_1 + velocities_2)

      # Update location orientation 
      for k, b in enumerate(self.bodies):
        b.location_new = b.location_old + velocities_new[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities_new[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation_old

      # Call postprocess
      postprocess_result = self.postprocess(self.bodies)

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        for k, b in enumerate(self.bodies):
          b.location = b.location_new
          b.orientation = b.orientation_new
        return
      else:
        for b in self.bodies:
          b.location = b.location_old
          b.orientation = b.orientation_old

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return



  def stochastic_Slip_Mid(self, dt): 
    ''' 
    Take a time step of length dt using a stochastic 
    mid-point method. The thermal drift is handle
    with a slip method.

    The computational cost of this scheme is 3 rigid solves
    + 2 lanczos call + 2 blob mobility product + 2 products with the geometri matrix K.

    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Call preprocess
      preprocess_result = self.preprocess(self.bodies)
      
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
        b.orientation_old = b.orientation

      # Generate random vector
      W1 = np.random.normal(0.0, 1.0, self.Nblobs*3)
      W_slip = np.random.normal(0.0, 1.0, self.Nblobs*3)
      Wcor = W1 + np.random.normal(0.0, 1.0, self.Nblobs*3)     

      # Compute M at time level n
      r_vectors_blobs_n = self.get_blobs_r_vectors(self.bodies, self.Nblobs)
      
      #compute M*W to be used by the corrector step
      MxW_slip = self.mobility_vector_prod(r_vectors_blobs_n, W_slip, self.eta, self.a, periodic_length = self.periodic_length)
      #compute K^T*W to be used by the corrector step
      KTxW_slip = self.K_matrix_T_vector_prod(self.bodies,W_slip, self.Nblobs)
      
      # Build preconditioners
      PC_partial, mobility_pc_partial, P_inv_mult = self.build_block_diagonal_preconditioners_det_stoch(self.bodies, 
                                                                                                        r_vectors_blobs_n, 
                                                                                                        self.Nblobs, 
                                                                                                        self.eta, 
                                                                                                        self.a,
                                                                                                        periodic_length=self.periodic_length)
      
      
      # Calc noise contributions M^{1/2}*W1 and M^{1/2}*(W1+W3)
      velocities_noise_W1, it_lanczos = stochastic.stochastic_forcing_lanczos(factor = np.sqrt(4*self.kT / dt),
                                                                              tolerance = self.tolerance, 
                                                                              dim = self.Nblobs * 3, 
                                                                              mobility_mult = mobility_pc_partial,
                                                                              L_mult = P_inv_mult,
                                                                              z = W1)
      
      velocities_noise_Wcor, it_lanczos = stochastic.stochastic_forcing_lanczos(factor = np.sqrt(self.kT / dt),
                                                                                tolerance = self.tolerance, 
                                                                                dim = self.Nblobs * 3, 
                                                                                mobility_mult = mobility_pc_partial,
                                                                                L_mult = P_inv_mult,
                                                                                z = Wcor)                                                                           
      
      # Solve mobility problem
      sol_precond = self.solve_mobility_problem(noise = velocities_noise_W1, 
                                                x0 = self.first_guess, 
                                                save_first_guess = True, 
                                                PC_partial = PC_partial)
      # Extract velocities
      velocities_mid = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))
      
      # Solve mobility problem
      slip_precond_rfd = self.solve_mobility_problem(RHS = np.concatenate([-1.0*W_slip, np.zeros(len(self.bodies) * 6)]), 
                                                     PC_partial = PC_partial)
      # Extract velocities
      W_RFD = np.reshape(slip_precond_rfd[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))
      
      # Update configuration for rfd 
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + W_RFD[k*6 : k*6+3] * self.rf_delta 
        quaternion_dt = Quaternion.from_rotation(W_RFD[(k*6+3):(k*6+6)] * self.rf_delta )
        b.orientation = quaternion_dt * b.orientation_old
        
      # Compute M at RFD time level
      r_vectors_blobs_rfd = self.get_blobs_r_vectors(self.bodies, self.Nblobs)
      #compute M*W to be used by the corrector step
      M_rfdxW_slip = self.mobility_vector_prod(r_vectors_blobs_rfd, W_slip, self.eta, self.a, periodic_length = self.periodic_length)
      #compute K^T*W to be used by the corrector step
      KT_rfdxW_slip = self.K_matrix_T_vector_prod(self.bodies,W_slip, self.Nblobs)
      
      rand_slip_cor = velocities_noise_Wcor + (self.kT / self.rf_delta)* (M_rfdxW_slip - MxW_slip)
      rand_force_cor = -1.0*(self.kT / self.rf_delta)*(KT_rfdxW_slip - KTxW_slip)
      
      # Update location orientation to mid point
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + velocities_mid[k*6 : k*6+3] * dt * 0.5
        quaternion_dt = Quaternion.from_rotation(velocities_mid[(k*6+3):(k*6+6)] * dt * 0.5)
        b.orientation = quaternion_dt * b.orientation_old

      # Solve mobility problem at the corrector step
      sol_precond_cor = self.solve_mobility_problem(noise = rand_slip_cor, 
                                                    noise_FT = rand_force_cor,
                                                    x0 = self.first_guess, 
                                                    save_first_guess = True,
                                                    PC_partial = PC_partial)

      # Extract velocities
      velocities_new = np.reshape(sol_precond_cor[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))     
      
      # Update location orientation 
      for k, b in enumerate(self.bodies):
        b.location_new = b.location_old + velocities_new[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities_new[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation_old
        
      # Call postprocess
      postprocess_result = self.postprocess(self.bodies)

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        for k, b in enumerate(self.bodies):
          b.location = b.location_new
          b.orientation = b.orientation_new
        return
      else:
        for b in self.bodies:
          b.location = b.location_old
          b.orientation = b.orientation_old

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return

  def stochastic_Slip_Mid_DLA(self, dt): 
    ''' 
    Take a time step of length dt using a stochastic
    first order Randon Finite Difference (RFD) schame.
    This method uses dense algebra methods.

    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Call preprocess
      preprocess_result = self.preprocess(self.bodies)
      
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
        b.orientation_old = b.orientation

      # Solve mobility problem predictor step
      velocities_mid, mobility_bodies_mid, mobility_blobs_mid, resistance_blobs_mid, K_mid, r_vectors_blobs_mid = self.solve_mobility_problem_DLA()
	
      # generate all of the necisarry random increments for the predictor step
      W1 = np.random.normal(0.0, 1.0, self.Nblobs*3)
      W_slip = np.random.normal(0.0, 1.0, self.Nblobs*3)
      Wcor = W1 + np.random.normal(0.0, 1.0, self.Nblobs*3)
        
      W_RFD = np.dot(mobility_bodies_mid,
			  np.dot(K_mid.T,
			  np.dot(resistance_blobs_mid,
			  W_slip)))

      # Compute K^{T}*W2 and M*W2 to be used by the corrector step
      MxW_slip = np.dot(mobility_blobs_mid,W_slip)
      KTxW_slip = np.dot(K_mid.T,W_slip)
      
      # Calculate relevant bit of stochastic increments at time level n
      Mhalf_W1 = stochastic.stochastic_forcing_eig_symm(mobility_blobs_mid, factor = 1.0, z = W1)
      Mhalf_Wcor = stochastic.stochastic_forcing_eig_symm(mobility_blobs_mid, factor = 1.0, z = Wcor)
        
      # Compute c1*N*K^T*M^(-1)*(c1*W2 + M^(1/2)*W1) for pred. step
      RHS_pred = np.sqrt(4*self.kT / dt)*np.dot(mobility_bodies_mid,
						  np.dot(K_mid.T,
						  np.dot(resistance_blobs_mid,
						  (Mhalf_W1))))
      
      # Compute pred. step velocities
      velocities_mid += RHS_pred
      
      # Compute RFD bits
      for k, b in enumerate(self.bodies):
	b.location = b.location_old + W_RFD[6*k:6*k+3] * self.rf_delta
	quaternion_dt = Quaternion.from_rotation(W_RFD[6*k+3:6*k+6] * self.rf_delta)
	b.orientation = quaternion_dt * b.orientation_old
	
      r_vectors_blobs_RFD = self.get_blobs_r_vectors(self.bodies, self.Nblobs)
      # Calculate mobility (M) at the blob level
      mobility_blobs_RFD = self.mobility_blobs(r_vectors_blobs_RFD, self.eta, self.a)
      # Calculate block-diagonal matrix K
      K_RFD = self.calc_K_matrix(self.bodies, self.Nblobs)
	
      DxM = np.dot(mobility_blobs_RFD,W_slip) - MxW_slip
      DxKT = np.dot(K_RFD.T,W_slip) - KTxW_slip
        

      for k, b in enumerate(self.bodies):
	b.location = b.location_old + velocities_mid[6*k:6*k+3] * dt * 0.5
	quaternion_dt = Quaternion.from_rotation((velocities_mid[6*k+3:6*k+6]) * dt * 0.5)
	b.orientation = quaternion_dt * b.orientation_old
        
      # Solve mobility problem predictor step 
      velocities_new, mobility_bodies_new, mobility_blobs_new, resistance_blobs_new, K_new, r_vectors_blobs_new = self.solve_mobility_problem_DLA()
        
        
      # Compute RHS of cor step so that N*RHS_cor is the correct increment
      RHS_cor = -(self.kT / self.rf_delta)*DxKT + np.dot(K_new.T,np.dot(resistance_blobs_new,
										np.sqrt(self.kT / dt)*Mhalf_Wcor
										+(self.kT / self.rf_delta)*DxM))
        
      # Compute cor. step velocities
      velocities_new += np.dot(mobility_bodies_new,RHS_cor)
      
        
      # Update location orientation to end point
      for k, b in enumerate(self.bodies):
	b.location_new = b.location_old + velocities_new[6*k:6*k+3] * dt
	quaternion_dt = Quaternion.from_rotation((velocities_new[6*k+3:6*k+6]) * dt)
	b.orientation_new = quaternion_dt * b.orientation_old

      # Call postprocess
      postprocess_result = self.postprocess(self.bodies)

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        for k, b in enumerate(self.bodies):
          b.location = b.location_new
          b.orientation = b.orientation_new
        return
      else:
        for b in self.bodies:
          b.location = b.location_old
          b.orientation = b.orientation_old

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return


  def solve_mobility_problem(self, RHS = None, noise = None, noise_FT = None, AB = None, x0 = None, save_first_guess = False, PC_partial = None): 
    ''' 
    Solve the mobility problem using preconditioned GMRES. Compute 
    velocities on the bodies subject to active slip and enternal 
    forces-torques.

    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      System_size = self.Nblobs * 3 + len(self.bodies) * 6

      # Get blobs coordinates
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      # If RHS = None set RHS = [slip, -force_torque]
      if RHS is None:
        # Calculate slip on blobs
        if self.calc_slip is not None:
          slip = self.calc_slip(self.bodies, self.Nblobs)
        else:
          slip = np.zeros((self.Nblobs, 3))
        # Calculate force-torque on bodies
        force_torque = self.force_torque_calculator(self.bodies, r_vectors_blobs)
        # Add noise to the force/torque
        if noise_FT is not None:
          force_torque += noise_FT
        # Set right hand side
        RHS = np.reshape(np.concatenate([slip, -force_torque]), (System_size))

      # Add noise to the slip
      if noise is not None:
        RHS[0:r_vectors_blobs.size] -= noise

      # Calculate K matrix
      K = self.calc_K_matrix_bodies(self.bodies, self.Nblobs)

      # Set linear operators 
      linear_operator_partial = partial(self.linear_operator, 
                                        bodies = self.bodies, 
                                        r_vectors = r_vectors_blobs, 
                                        eta = self.eta, 
                                        a = self.a, 
                                        K_bodies = K,
                                        periodic_length=self.periodic_length)
      A = spla.LinearOperator((System_size, System_size), matvec = linear_operator_partial, dtype='float64')

      # Set preconditioner 
      if PC_partial is None:
        PC_partial = self.build_block_diagonal_preconditioner(self.bodies, r_vectors_blobs, self.Nblobs, self.eta, self.a)
      PC = spla.LinearOperator((System_size, System_size), matvec = PC_partial, dtype='float64')
      
      # Scale RHS to norm 1
      RHS_norm = np.linalg.norm(RHS)
      if RHS_norm > 0:
        RHS = RHS / RHS_norm

      # Solve preconditioned linear system # callback=make_callback()
      (sol_precond, info_precond) = spla.gmres(A, RHS, x0=x0, tol=self.tolerance, M=PC, maxiter=1000, restart=60) 
      if save_first_guess:
        self.first_guess = sol_precond  

      # Scale solution with RHS norm
      if RHS_norm > 0:
        sol_precond = sol_precond * RHS_norm
      
      # Return solution
      return sol_precond


  def solve_mobility_problem_dense_algebra(self): 
    ''' 
    Solve the mobility problem using dense algebra methods. Compute 
    velocities on the bodies subject to active slip and enternal 
    forces-torques.
    
    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Calculate slip on blobs
      if self.calc_slip is not None:
        slip = self.calc_slip(self.bodies, self.Nblobs)
      else:
        slip = np.zeros((self.Nblobs, 3))

      # Get blobs coordinates
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      # Calculate mobility (M) at the blob level
      mobility_blobs = self.mobility_blobs(r_vectors_blobs, self.eta, self.a)

      # Calculate resistance at the blob level (use np.linalg.inv or np.linalg.pinv)
      resistance_blobs = np.linalg.inv(mobility_blobs)

      # Calculate constraint force due to slip l = M^{-1}*slip
      force_slip = np.dot(resistance_blobs, np.reshape(slip, (3*self.Nblobs,1)))

      # Calculate force-torque on bodies
      force_torque = self.force_torque_calculator(self.bodies, r_vectors_blobs)

      # Add slip force looping over bodies
      offset = 0
      for k, b in enumerate(self.bodies):
        K = b.calc_K_matrix()
        force_torque[2*k : 2*(k+1)] -= np.reshape(np.dot(K.T, force_slip[3*offset : 3*(offset+b.Nblobs)]), (2, 3))
        offset += b.Nblobs    

      # Calculate block-diagonal matrix K
      K = self.calc_K_matrix(self.bodies, self.Nblobs)
    
      # Calculate mobility (N) at the body level. Use np.linalg.inv or np.linalg.pinv
      mobility_bodies = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)), rcond=1e-14)

      # Compute velocities, return velocities and bodies' mobility
      return (np.dot(mobility_bodies, np.reshape(force_torque, 6*len(self.bodies))), mobility_bodies)


  def solve_mobility_problem_DLA(self): 
    ''' 
    Solve the mobility problem using dense algebra methods. Compute 
    velocities on the bodies subject to active slip and enternal 
    forces-torques.
    
    The linear and angular velocities are sorted like
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Calculate slip on blobs
      if self.calc_slip is not None:
        slip = self.calc_slip(self.bodies, self.Nblobs)
      else:
        slip = np.zeros((self.Nblobs, 3))

      # Get blobs coordinates
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      # Calculate mobility (M) at the blob level
      mobility_blobs = self.mobility_blobs(r_vectors_blobs, self.eta, self.a)

      # Calculate resistance at the blob level (use np.linalg.inv or np.linalg.pinv)
      resistance_blobs = np.linalg.inv(mobility_blobs)

      # Calculate block-diagonal matrix K
      K = self.calc_K_matrix(self.bodies, self.Nblobs)
     
      # Calculate constraint force due to slip l = M^{-1}*slip
      force_slip = np.dot(K.T,np.dot(resistance_blobs, np.reshape(slip, (3*self.Nblobs,1))))
      
      # Calculate force-torque on bodies
      force_torque = self.force_torque_calculator(self.bodies, r_vectors_blobs)
      
      # Calculate RHS
      FT = np.reshape(force_torque, 6*len(self.bodies))
      FTS = FT + np.reshape(force_slip, 6*len(self.bodies))

      # Calculate mobility (N) at the body level. Use np.linalg.inv or np.linalg.pinv
      mobility_bodies = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)), rcond=1e-14)

      # Compute velocities
      return (np.dot(mobility_bodies, FTS), mobility_bodies, mobility_blobs, resistance_blobs, K, r_vectors_blobs)


# Callback generator
def make_callback():
  closure_variables = dict(counter=0, residuals=[]) 
  def callback(residuals):
    closure_variables["counter"] += 1
    closure_variables["residuals"].append(residuals)
    print closure_variables["counter"], residuals
  return callback


