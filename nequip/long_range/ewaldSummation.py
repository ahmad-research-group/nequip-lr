import torch
import math
from nequip.data import AtomicDataDict
import numpy as np
from nequip.long_range.HirshFeld import *
# sigmadata = torch.tensor([1.,2.,3.])
'''
Eelec = Ereal + Ereceip + Eself

Ereal = 1/2 * sum(i = 1 ~ Nat) { sum(j=/ i to Nneigh) { Qi * Qj * erfc(rij/sqrt(2)/eta) / rij } }

Ereceip = 2*pi / V * sum(k =/ 0) {exp(-eta**2 |k|**2)/2} / |k|**2 * |S(k)|**2

S(k) = sum(i=1 to Nat) {Qi exp(i*k . ri)}

Eself = - sum(i=1 to Nat) {Qi**2}/ sqrt(2*pi)/ eta

# '''

# data = {
#      'pos': torch.tensor([[2.7320, 4.3236, 3.5625],
#         [0.7503, 1.2541, 3.7962],
#         [1.0590, 4.2208, 5.0442],
#         [2.9842, 5.0778, 5.0649],
#         [1.3311, 4.7296, 3.2238],
#         [4.5336, 4.5982, 1.7281],
#         [0.0900, 4.9429, 2.9770],
#         [1.3005, 2.9135, 4.8747]], dtype = float) ,
#         'initial_charges' : [0.5,1,1,1,-1,-1,-1,-0.5],
#         'sigma' : torch.tensor([2,2,2,2,1,1,1,1]),
#         'cell' : [[5.0842, 0.0000, 0.0000],
#          [0.0000, 5.0842, 0.0000],
#          [0.0000, 0.0000, 5.0842]]

# }

k_cutoff = 5.0
def build_sigma(atoms):
    sigma = []
    # atoms_type = {
    #     '0': 0.2,
    #     '1': 0.1,
    #     '2': 0.3,
    #     '3': 0.5
    # }

    # O, Mg, Al, Au -> sorted in alphabetic order
    atoms_type = [0.6, 1.41, 1.21, 1.36]
    for atom in atoms:
        sigma.append(atoms_type[atom])
    # print(sigma)
    return torch.tensor(sigma)

def build_A(J, sigma,r,gamma,Nat):
    #modify A for langrange multiplier
    A = torch.randn(Nat, Nat)
    for i in range(Nat):
        for j in range(Nat):
            if(i==j):
                # print(i,j)
                A[i][j] = J[i]+ 1/(sigma[i]*torch.sqrt(torch.tensor(3.1416)))
            else:
                # print('rij',r[i][j])
                A[i][j] = torch.erfc(r[i][j]/(torch.tensor(1.4142)*gamma[i][j]))/r[i][j]
            # if torch.isnan(A[i][j]) or torch.isinf(A[i][j]):
            #     print('i, j = ', i, j)
            #     print(r[i][j])
    # print("A",A.flatten())
    # rows, cols = A.shape
    # row_ones = torch.ones(1, cols)
    # col_ones = torch.ones(rows + 1, 1)
    # tensor_with_row = torch.cat((A, row_ones), dim=0)
    # tensor_with_row_and_col = torch.cat((tensor_with_row, col_ones), dim=1)
    # tensor_with_row_and_col[-1,-1] = 0
    # print("A after changes",tensor_with_row_and_col.flatten())
    # return tensor_with_row_and_col
    return A

def getHfCharges(X,A):
    # if(len(X)>8):
    #     return 0
    # X = data['charges']
    # pos = data['pos']
    # Nat = len(pos)
    # J = torch.randn(Nat, requires_grad=True)
    # gamma = build_gamma(sigma, Nat)
    # r = build_r(pos, r_max)
    # A = build_A(J, sigma, r, gamma, Nat)
    # print(A.shape)
    # A = A.detach().numpy()
    # X = X.detach().numpy()
    # print(A)
    # print(X)
    # X = np.transpose(X)
    #torch.linalg.solve
    Q = np.matmul(np.linalg.inv(A),-X) 
    # Q = torch.linalg.solve(A,-X) 
    # if(np.isnan(Q.any())):
    #     return 0
    return torch.tensor(Q, requires_grad=True) 

def build_gamma(sigma,Nat):
    
    gamma = torch.zeros((Nat, Nat))
    for i in range(Nat):
        for j in range(Nat):
            gamma[i][j] = torch.sqrt(sigma[i]**2 + sigma[j]**2)
            
    return gamma       


#interatomic distances matrix
def build_r(pos, r_max):
    # Ensure pos is a numpy array
    pos = pos.clone().detach().numpy()

    # Calculate interatomic distances
    n_atoms = len(pos)
    r = torch.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            distance = torch.tensor(np.linalg.norm(pos[i] - pos[j]))
            if distance <= r_max:
                r[i][j] = distance
                r[j][i] = distance

    return torch.tensor(r)

#point charges
def ewaldSummationPC(Q, pos, cell, Nat,r):
    
    ''' 
    calculates real part of ewald Summation for point charges 
    
    args:
        charges[], positions[], neighboring_atoms[Nneigh]
        
    '''

    # eta is the standard deviation of the Gaussian charges, which are placed on the point charges to remove the long-range interactions.
    eta = torch.tensor(0.005) #check fortran code
    def ewaldReal(Nat, eta,Q):
        Ereal = torch.tensor(0.0)
        for i in range(Nat):
            for j in range(Nat):
                if(i==j):
                    continue
                rij = r[i][j]
                # print("rij",rij)
                # print(Q[i],Q[j])
                Ereal += Q[i][0] * Q[j][0] * torch.erfc(torch.tensor(rij) / (torch.sqrt(torch.tensor(2.0)) * eta)) / rij
        Ereal *= 0.5
        return Ereal


    ''' calculates receiprocal part of ewald Summation for point charges '''
    def ewaldReceip(cell_vectors, k_cutoff):

        #generating k-points
        def generate_k_points(cell, k_cutoff):
            # Calculate the reciprocal lattice vectors
            reciprocal_vectors = 2 * np.pi * np.linalg.inv(cell).T


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
            return torch.tensor(k_points,dtype = float)           

        k_points = generate_k_points(cell, k_cutoff)
        # print(k_points)
        Ereceip = 0.0
        for k_val in k_points:
            k_mag_sq = torch.dot(k_val, k_val)
            if torch.abs(k_mag_sq)>1e-6:    
                Sk = torch.tensor(0)
                for i in range(Nat):
                    Sk = Sk + Q[i] * torch.exp(1j * torch.dot(k_val.float(), pos[i,:].float()))
                Ereceip += torch.exp(-eta**2 * k_mag_sq / 2) / k_mag_sq * torch.abs(Sk)**2
                # print("k",k_mag_sq * torch.abs(Sk)**2)
        V = torch.linalg.det(torch.tensor(cell))
        # print("V",V)
        Ereceip *= 2 * math.pi / V
        return Ereceip


    ''' calculates self part of ewald Summation for point charges '''
    def ewaldSelf(Q,eta):
        Eself = 0
        for q in Q:
            Eself += -q**2 / (math.sqrt(2 * math.pi) * eta)
        return torch.tensor(Eself)

    realPart = ewaldReal(Nat, eta,Q)
    receipPart = ewaldReceip(cell, k_cutoff)
    selfPart = ewaldSelf(Q,0.005)
    # print(selfPart)
    ewaldSumPC = realPart + receipPart + selfPart

    return ewaldSumPC


def ewaldSummationGauss(pos, charges,Nat,r,gamma,cell,sigma):
    Q = charges
    energyPC = ewaldSummationPC(Q, pos,cell,Nat,r)
    def calc_subPart(charges, pos, gamma, sigma):
        energy = torch.tensor([0.0])  # Initialize energy as a tensor
        # First term of the equation
        for i in range(Nat):
            for j in range(Nat):
                if i != j:
                    energy += Q[i] * Q[j] * torch.erfc(r[i][j] / torch.tensor(np.sqrt(2) * gamma[i][j])) / r[i][j]
        # Second term of the equation
        for i in range(Nat):
            energy += Q[i]**2 / (2 * np.sqrt(np.pi) * sigma[i])
        return energy


    subtraction_part = calc_subPart(Q, pos, gamma, sigma)

    energyGaussian = energyPC - subtraction_part
    return energyGaussian

def ewaldSummation(data):
    # if(len(data['pos']))>110:
    #     return 0
    '''
    Args:
    output data[]

    '''
    # atom_types = data['atom_types'].shape
    r_max = 10
    # X = torch.rand(atom_types, requires_grad = True)
    # J = torch.rand(atom_types, requires_grad = True)
    
    
    X = data['initial_charges'].clone().detach().numpy()
    # print(X1)
    # zero = torch.tensor([[1e-6]])
    # print(zero.shape)
    # X = torch.cat((X1,zero))
    
    pos = data['pos']
    atoms = data['atom_types'].flatten()
    # sigma = sigmadata[data[AtomicDataDict.ATOM_TYPE_KEY]]
    # if (len(X)!=0)
    
    # print(Q)
    
    # print('Q',Q)
    Nat = len(pos)
    # if(Nat>110): 
    #     return 0
    r = build_r(pos, r_max)
    # sigma = torch.tensor([0.2,0.2,0.2,0.2,0.1,0.1,0.1,0.1]) #gaussian distribution with width sigma(i)
    sigma = build_sigma(atoms)
    gamma = build_gamma(sigma,Nat)
    cell = data['cell'][0]
    J = torch.randn(Nat,requires_grad=True)
    # J = J
    A = build_A(J,sigma,r,gamma,Nat)
    # X = data['initial_charges'].detach().numpy()
    A = A.clone().detach().numpy()
    # A = data['initial_charges'].detach().numpy()
    # print('X',len(X))
    # print('A',len(A))
    Q = getHfCharges(X, A)
    # Q = Q[:Nat]
    # print(Q)
    # print(Nat)
    # if isinstance(Q, int):
    #     return 0
    # print('Q sum =',Q.sum())
    elec = ewaldSummationGauss(pos,Q,Nat,r,gamma,cell,sigma)
    # print('electrostatic part',elec)

    # if(torch.isnan(elec)): # do not constrain
    #     return 0
    # else:
    #     return elec
    return elec
# ans = ewaldSummation(data)
# print(ans)
