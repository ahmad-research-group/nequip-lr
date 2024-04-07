import torch
import math
from nequip.data import AtomicDataDict
import numpy as np

print(AtomicDataDict.POSITIONS_KEY)

'''
Eelec = Ereal + Ereceip + Eself

Ereal = 1/2 * sum(i = 1 ~ Nat) { sum(j=/ i to Nneigh) { Qi * Qj * erfc(rij/sqrt(2)/eta) / rij } }

Ereceip = 2*pi / V * sum(k =/ 0) {exp(-eta**2 |k|**2)/2} / |k|**2 * |S(k)|**2

S(k) = sum(i=1 to Nat) {Qi exp(i*k . ri)}

Eself = - sum(i=1 to Nat) {Qi**2}/ sqrt(2*pi)/ eta

'''

def ewaldSummationPC(Q, pos, neighboring_atoms):

    ''' 
    calculates real part of ewald Summation for point charges 
    
    args:
        charges[], positions[], neighboring_atoms[Nneigh]
    
    '''
    def ewaldReal():
        Ereal = 0.0
        for i in range(len(Q)):
            for j in range(Nneigh):
                if(i==j):
                    continue
                rij = r[i, j]
                Ereal += Q[i] * Q[j] * torch.erfc(rij / math.sqrt(2) / eta) / rij
        Ereal *= 0.5
        return Ereal


    ''' calculates receiprocal part of ewald Summation for point charges '''
    def ewaldReceip():

        import numpy as np

        def generate_k_points(cell_vectors, k_cutoff):
            # Calculate the reciprocal lattice vectors
            reciprocal_vectors = 2 * np.pi * np.linalg.inv(cell_vectors).T
            
            # Calculate the maximum index for each dimension
            max_index_x = int(np.ceil(np.sqrt(k_cutoff) / np.linalg.norm(reciprocal_vectors[0])))
            max_index_y = int(np.ceil(np.sqrt(k_cutoff) / np.linalg.norm(reciprocal_vectors[1])))
            max_index_z = int(np.ceil(np.sqrt(k_cutoff) / np.linalg.norm(reciprocal_vectors[2])))
            
            # Create meshgrids for each dimension
            indices_x = np.arange(-max_index_x, max_index_x + 1)
            indices_y = np.arange(-max_index_y, max_index_y + 1)
            indices_z = np.arange(-max_index_z, max_index_z + 1)
            mesh_x, mesh_y, mesh_z = np.meshgrid(indices_x, indices_y, indices_z, indexing='ij')
            
            mesh_flat = np.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()], axis=-1)
            k_points = np.dot(mesh_flat, reciprocal_vectors)
            
            # Filter k-points within the Brillouin zone based on k_cutoff
            k_norms = np.linalg.norm(k_points, axis=1)
            k_points = k_points[k_norms <= k_cutoff]
            
            return k_points

            #cell_vectors = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])  
            #k_cutoff = 5.0  

            #k_points = generate_k_points(cell_vectors, k_cutoff)
            #print(k_points)


        def calculate_Sk(Q, r, k):

            '''
            args : charges[], positions[], k_points[], r[]

            '''

            Sk = torch.zeros_like(k, dtype=torch.complex64)
            for i in range(len(Q)):
                exp_term = torch.exp(1j * torch.dot(k, r[i]))
                Sk += Q[i] * exp_term
            return Sk
        
        
        Ereceip = 0.0
        for k_val in k_points:
            if not torch.all(k_val == 0):
                k_mag_sq = torch.dot(k_val, k_val)
                Sk = calculate_Sk(Q, r, k_val)
                Ereceip += torch.exp(-eta**2 * k_mag_sq / 2) / k_mag_sq * torch.abs(Sk)**2
        Ereceip *= 2 * math.pi / V
        return Ereceip


    ''' calculates self part of ewald Summation for point charges '''
    def ewaldSelf():
        Eself = torch.sum(-Q**2) / (math.sqrt(2 * math.pi) * eta)
        return Eself

    realPart = ewaldReal()
    receipPart = ewaldReceip()
    selfPart = ewaldSelf()
    
    ewaldSumPC = realPart + receipPart + selfPart

    return ewaldSumPC


def ewaldSummationGauss():

    energyPC = ewaldSummationPC()
    subtraction_part = ...
    energyGaussian = energyPC - subtraction_part

    return energyGaussian


