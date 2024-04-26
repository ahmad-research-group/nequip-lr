from ase.io import *

from nequip.ase import NequIPCalculator

atomslist = read('output.extxyz',':')

atoms = atomslist[0]
# print(atoms.arrays['initial_charges'])

atoms.calc = NequIPCalculator.from_deployed_model(
    model_path="deployed.pth",
    species_to_type_name = {
        "Li": "Li",
        "Cl": "Cl"
    }
)
energy = atoms.get_potential_energy()
# print(energy)
# print(atoms.info['energy'])


# print(atoms.arrays['initial_charges'])
#print(atoms.get_charges())
#    print(energies)



#print(energies)
