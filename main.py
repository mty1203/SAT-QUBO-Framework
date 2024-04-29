from sat.models.prototypes.nmprototypesat import NMPrototypeSAT
from sat.qubosearcher import *
from sat.formula_creator import *
import os
from sat.models.choi import ChoiSAT
from sat.models.chancellor import ChancellorSAT
from sat.models.nuesslein import NuessleinNM

import numpy as np


def ising_machine_with_perturbation(ising_matrix, max_iterations=10000, energy_threshold=None, initial_temperature=1.0,
                                    final_temperature=0.01):
   # Initialize the state vector S with random values of -1 or +1
   S = np.random.choice([-1, 1], size=ising_matrix.shape[0])

   # Normalize the Ising matrix to prevent large values that could cause numerical instability
   # Here we simply scale it to have its largest absolute eigenvalue be 1
   max_eigenvalue = np.max(np.abs(np.linalg.eigvals(ising_matrix)))
   normalized_matrix = ising_matrix / max_eigenvalue if max_eigenvalue != 0 else ising_matrix

   # Store the best energy and state found
   best_energy = float('inf')
   best_state = None

   # Temperature factor for perturbation
   temperature = initial_temperature

   # Temperature decay factor
   decay = (final_temperature / initial_temperature) ** (1 / max_iterations)
   index = 0
   # Perform the iterative process
   for iteration in range(max_iterations):
      # Compute the product MxS
      MS = normalized_matrix @ S
      index += 1
      # Calculate the energy of the current state
      energy = -S @ MS / 2  # Since E = -1/2 * S' * M * S

      # Update the best energy and state if the current one is better
      if energy < best_energy:
         best_energy = energy
         best_state = S.copy()

      # If energy threshold is defined and the current energy is below it, break the loop
      if energy_threshold is not None and energy < energy_threshold:
         break

      # Update the state vector S with a chance of random perturbation
      for i in range(len(S)):
         # Calculate the energy difference if S[i] were flipped
         delta_energy = 2 * S[i] * MS[i]

         # Flip the spin if the new state has lower energy, or with a probability related to the "temperature"
         if delta_energy > 0 or np.exp(delta_energy / temperature) > np.random.rand():
            S[i] *= -1

      # Decrease the temperature (reduce the perturbation over time)
      temperature *= decay
   print('max iterations:', iteration)
   return best_state, best_energy


# Apply the Ising machine algorithm with perturbation to the normalized Ising matrix


# Display the best state and energy found by the Ising machine with perturbation



def ising_machine(ising_matrix, max_iterations=100000000, energy_threshold=None):
   # Initialize the state vector S with random values of -1 or +1
   S = np.random.choice([-1, 1], size=ising_matrix.shape[0])

   # Store the best energy and state found
   best_energy = float('inf')
   best_state = None
   index = 0
   # Perform the iterative process
   for iteration in range(max_iterations):
      index=index+1
      # Compute the product MxS
      MS = ising_matrix @ S

      # Calculate the energy of the current state
      energy = -S @ MS / 2  # Since E = -1/2 * S' * M * S

      # Update the best energy and state if the current one is better
      if energy < best_energy:
         best_energy = energy
         best_state = S.copy()

      # If energy threshold is defined and the current energy is below it, break the loop
      if energy_threshold is not None and energy < energy_threshold:
         break

      # Update the state vector S
      for i in range(len(S)):
         # Calculate the energy difference if S[i] were flipped
         delta_energy = 2 * S[i] * MS[i]

         # Flip the spin if the new state has lower energy
         if delta_energy > 0:
            S[i] *= -1
   print('max index = {}'.format(index))
   return best_state, best_energy


if __name__ == '__main__':

   #
   # -- Copy examples from examples.py to get started
   #

   formula = load_formula_from_dimacs_file(os.path.join(os.getcwd(), "formulas", "sat_k3_v64_c192.cnf"))
   print(formula)
   # choi_sat = ChoiSAT(formula)
   # choi_sat.create_qubo()
   nuesslein_sat = NuessleinNM(formula)
   nuesslein_sat.create_qubo()
   qubo_dict = nuesslein_sat.qubo
   max_index = max(max(pair) for pair in qubo_dict.keys())
   size = max_index + 1

# Create an empty matrix of the appropriate size
   qubo_matrix = np.zeros((size, size))

# Populate the QUBO matrix using the dictionary
   for (i, j), value in qubo_dict.items():
       qubo_matrix[i, j] = value
       if i != j:  # Assume symmetry for the QUBO matrix
           qubo_matrix[j, i] = value

# Convert QUBO matrix to Ising model
# In the Ising model, s_i variables are -1 or +1, so we need to transform the QUBO variables (0 or 1) accordingly
# The transformation formula is s = 2x - 1, where x is the QUBO variable and s is the Ising variable
   ising_matrix = qubo_matrix.copy()
   np.fill_diagonal(ising_matrix, qubo_matrix.diagonal() - np.sum(qubo_matrix, axis=1))
   ising_matrix /= 4

# The off-diagonal elements are halved in the transformation from QUBO to Ising (because Q_ij = J_ij/4)
   for i in range(size):
      for j in range(size):
         if i != j:
             ising_matrix[i, j] /= 2
   # best_state, best_energy = ising_machine(ising_matrix)
   best_state, best_energy = ising_machine_with_perturbation(ising_matrix)
   print(best_state,len(best_state), best_energy)