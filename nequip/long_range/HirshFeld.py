import torch
# from nequip.data import AtomicData, AtomicDataDict
import numpy as np
from ase.neighborlist import get_distance_matrix

from nequip.data.AtomicData import neighbor_list_and_relative_vec

#dummy values to test the code
NSpecies = 2
Nat = 8
pos = [[2.7320, 4.3236, 3.5625],
        [0.7503, 1.2541, 3.7962],
        [1.0590, 4.2208, 5.0442],
        [2.9842, 5.0778, 5.0649],
        [1.3311, 4.7296, 3.2238],
        [4.5336, 4.5982, 1.7281],
        [0.0900, 4.9429, 2.9770],
        [1.3005, 2.9135, 4.8747]]

cell = [[[5.0842, 0.0000, 0.0000],
         [0.0000, 5.0842, 0.0000],
         [0.0000, 0.0000, 5.0842]]]

X = [0.98, 0.98, 0.98, 0.98, 3.16, 3.16, 3.16, 3.16]
Q = [1,1,1,1,-1,-1,-1,-1]
J = [0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2]
sigma = [2,2,2,2,1,1,1,1]

r_max = 10

# J = [Nat] = set_as_learnable parameter #Hardness
# X = [Nat] = set_as_learnable parameter #Chi - predicted Electronegativity
# Q = [Nat] = set_as_learnable parameter #Calculated charges
# r = [Nat][Nat] #interatomic distances

# sigma = [Nat] = set_as_learnable parameter

def getNeighAtoms():
    pass

def build_gamma(sigma):
    gamma = np.zeros((Nat, Nat))
    for i in range(Nat):
        for j in range(Nat):
            gamma[i][j] = np.sqrt(sigma[i]**2 + sigma[j]**2)
    return gamma        


def build_r(pos, r_max):
    # Ensure pos is a numpy array
    # pos = pos.detach().numpy()
    pos = torch.tensor(pos)
    
    # Calculate interatomic distances
    n_atoms = len(pos)
    r = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            distance = np.linalg.norm(pos[i] - pos[j])
            if distance <= r_max:
                r[i][j] = distance
                r[j][i] = distance
    
    return r


def getEelect(pos, r_max):
    r = build_r(pos,r_max)
    gamma = build_gamma(sigma)
    Eelec = 0
    for i in range(Nat):
        add_part = X[i]*Q[i] + 1/2*J[i]*Q[i]**2
        for j in range(i+1, Nat):
            rij = r[i][j]
            gammaij = gamma[i][j]
            Eelec = torch.erf(torch.tensor(rij/(np.sqrt(2)*gammaij)))*Q[i]*Q[j]/rij
        Eelec += Q[i]**2/(2*sigma[i]*np.sqrt(torch.pi))
    
    return Eelec+add_part
    
# [edge_index, shifts, cell_tensor]= neighbor_list_and_relative_vec(pos,1.4)


# distances = get_distance_matrix(pos,3)
# print(distances)

Elong =getEelect(pos,r_max)
print(Elong)


# from ase.neighborlist import NeighborList

# def build_distances(pos, r_max):
#     # Set up neighbor list
#     nl = NeighborList([r_max/2] * len(pos), self_interaction=False, bothways=True)
#     nl.update( pos)

#     # Initialize distances array
#     distances = np.zeros((len(pos), len(pos)))

#     # Iterate over atoms and calculate distances to neighbors
#     for i in range(len(pos)):
#         neighbor_indices, _ = nl.get_neighbors(i)
#         neighbor_positions = pos[neighbor_indices]
#         diff = neighbor_positions - pos[i]
#         distances[i, neighbor_indices] = np.linalg.norm(diff, axis=1)

#     return distances


# # Set the maximum distance for neighbor calculations
# r_max = 5.0  # Adjust as needed

# # Build the distances array using the function
# distances = build_distances(pos, r_max)

# print(distances)

# r = build_r(pos,r_max)
# print(r)