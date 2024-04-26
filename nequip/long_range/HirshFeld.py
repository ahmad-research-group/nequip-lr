import torch

NSpecies = ..
Nat = ...

J = [Nat] = set_as_learnable parameter #Hardness
X = [Nat] = set_as_learnable parameter #Chi - Electronegativity
Q = [Nat] = set_as_learnable parameter #predicted charges
r = [Nat][Nat] #interatomic distances

gamma = build_gamma()
sigma = [Nat] = set_as_learnable parameter

def getNeighAtoms:
    pass

def build_gamma:
    gamma = [Nat][Nat]
    for i in range(Nat):
        for j in range(Nat):
            gamma[i][j] = torch.sqrt(sigma[i]**2 + sigma[j]**2)
    return gamma        

def build_r:
    pass 

def getELong:
    Elong = 0
    for i in range(Nat):
        add_part = X[i]*Q[i] + 1/2*J[i]*Q[i]**2
        for j in range(Nneigh):
            rij = r[i][j]
            gammaij = gamma[i][j]
            Eelec = torch.erf(rij/(sqrt(2)*gammaij))*Q[i]*Q[j]/rij
        Eelec += Q[i]**2/(2*sigma[i]*sqrt(torch.pi))
    
    return Elong
     