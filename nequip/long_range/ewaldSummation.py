import torch
import math



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
                if(i==j) continue
                rij = r[i, j]
                Ereal += Q[i] * Q[j] * torch.erfc(rij / math.sqrt(2) / eta) / rij
        Ereal *= 0.5
        return Ereal


    ''' calculates receiprocal part of ewald Summation for point charges '''
    def ewaldReceip():

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
        for k_val in k:
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


