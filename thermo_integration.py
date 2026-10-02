from mchammer.free_energy_tools import get_free_energy_temperature_integration
from helper_functions import *
from numpy import mean
import matplotlib.pyplot as plt

from ase import Atoms
from icet import ClusterSpace
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalEnsemble
from mchammer.free_energy_tools import get_free_energy_thermodynamic_integration
from mchammer.ensembles import ThermodynamicIntegrationEnsemble
from mchammer.ensembles import CanonicalAnnealing


GGI_directories = [
                    "0.00000000000",
                    "0.11111111110",
                    "0.22222222220",
                    "0.33333333330",
                    "0.44444444440",
                    "0.55555555550",
                    "0.66666666660",
                    "0.77777777770",
                    "0.88888888880",
                    "1.00000000000",
                  ]

data_from_temperature_per_atom = []

cutoffs = [6, 4.5]


data_from_thermal = []
#data_from_thermal_per_atom = []


ATAT_format_lattice_path = "initial_data_files/lat.in"

cell, positions, chemical_symbols = parse_ATAT_lat(ATAT_format_lattice_path)

initial_symbols = [i[0] for i in chemical_symbols]

primitive = Atoms(
                   symbols = initial_symbols,
                   scaled_positions = positions,
                   cell = cell,
                   pbc = True
                 )

cluster_space = ClusterSpace(
                              primitive,
                              cutoffs = cutoffs,
                              chemical_symbols = chemical_symbols
                            )

cluster_expansion = ClusterExpansion.read("generated_data_files/cluster_expansion.ce")

supercell = primitive.repeat((5, 5, 5))
calc = ClusterExpansionCalculator(supercell, cluster_expansion)


mc_n_equilibration_steps = 10000
TDI_steps = 100000
annealing_n_steps = 100000


T_max = 1500
T_min = 100


k_B = 8.617333262e-5

start_configuration = supercell.copy()
sublattices = cluster_space.get_sublattices(start_configuration)
Ga_condition = False
Cu_condition = False 
Se_condition = False

for sublattice in sublattices:
    if 'Ga' in sublattice.chemical_symbols:
        ga_sites = sublattice.indices
        if Cu_condition and Se_condition:
            break
        Ga_condition = True     

    elif 'Cu' in sublattice.chemical_symbols:
        cu_sites = sublattice.indices
        if Ga_condition and Se_condition:
             break
        Cu_condition = True

    elif 'Se' in sublattice.chemical_symbols:
        se_sites = sublattice.indices
        if Ga_condition and Cu_condition:
            break
        Se_condition = True


for i in ga_sites:
    start_configuration[i].symbol = "In"

for i in cu_sites:
    start_configuration[i].symbol = "Cu"

for i in se_sites:
    start_configuration[i].symbol = "Se"

max_Ga_In_number = int(len(supercell) / 4)



Ga_number = len(ga_sites)//2
for i in ga_sites[0 : Ga_number]:
    start_configuration[i].symbol = 'Ga'
for i in ga_sites[Ga_number : len(ga_sites)]:
    start_configuration[i].symbol = 'In'

mc = CanonicalEnsemble(
    structure = start_configuration,
    calculator = calc,
    temperature = T_min,
    boltzmann_constant = k_B,
    trajectory_write_interval = None,
    ensemble_data_write_interval = 100)

mc.run(mc_n_equilibration_steps)

mc = ThermodynamicIntegrationEnsemble(
    structure = mc.structure,
    calculator=calc,
    temperature = T_min,
    forward = True,
    ensemble_data_write_interval = 1,
    boltzmann_constant = k_B ,
    n_steps = TDI_steps)
mc.run()

(temperatures_integration, free_energy_integration_forward) = \
        get_free_energy_thermodynamic_integration(mc.data_container, cluster_space,
                                                forward = True,
                                                max_temperature = T_max,
                                                boltzmann_constant = k_B)

plt.plot(temperatures_integration, free_energy_integration_forward, label = "TDI")

import random
random.seed(1)
ga_idx = random.sample(list(ga_sites), Ga_number)
for i in ga_sites:
    start_configuration[i].symbol = 'In'
for i in ga_idx:
    start_configuration[i].symbol = 'Ga'

mc = CanonicalEnsemble(
    structure = start_configuration,
    calculator = calc,
    temperature = T_max,
    boltzmann_constant = k_B,
    trajectory_write_interval = None,
    ensemble_data_write_interval = 1)

mc.run(mc_n_equilibration_steps)

Enthalpy_of_random = np.average(mc.data_container.get("potential")[1000:])
sublattices = cluster_space.get_sublattices(mc.structure)
sublattice_probabilities = [True] * len(sublattices)
from mchammer.free_energy_tools import _ideal_mixing_entropy, _get_atoms_on_sublattice
atoms_on_sublattices = _get_atoms_on_sublattice(mc.structure,
                                                sublattices)
ideal_mixing_entropy = _ideal_mixing_entropy(sublattice_probabilities,
                                             atoms_on_sublattices,
                                             k_B)
free_energy_reference = Enthalpy_of_random - ideal_mixing_entropy * T_max

print (free_energy_reference, Enthalpy_of_random, ideal_mixing_entropy)


mc = CanonicalAnnealing(
        structure  = mc.structure,
        calculator = calc,
        T_start = T_max,
        T_stop  = T_min,
        cooling_function = 'linear',
        n_steps = annealing_n_steps,
        boltzmann_constant = k_B,
        trajectory_write_interval = None,
        ensemble_data_write_interval = 1)

mc.run()


(temperatures_temperature, free_energy_temperature_forward) = \
    get_free_energy_temperature_integration(mc.data_container,
                                            cluster_space,
                                            forward = True,
                                            temperature_reference = T_max,
                                            free_energy_reference = free_energy_reference,
                                            max_temperature = T_max,
                                            boltzmann_constant = k_B)



plt.plot(temperatures_temperature, free_energy_temperature_forward, label = "Forward, TEI")
plt.plot(mc.data_container.data["temperature"], mc.data_container.data["potential"], label = "Forward, TEI")

mc = CanonicalAnnealing(
        structure = mc.structure,
        calculator = calc,
        T_start = T_min,
        T_stop = T_max,
        cooling_function = 'linear',
        n_steps = annealing_n_steps,
        boltzmann_constant = k_B,
        trajectory_write_interval = None,
        ensemble_data_write_interval = 1)

mc.run()

print (free_energy_integration_forward[0])
print (temperatures_integration[0])






(temperatures_temperature, free_energy_temperature_backward) = \
    get_free_energy_temperature_integration(mc.data_container,
                                            cluster_space,
                                            forward = False,
                                            temperature_reference = T_max,
                                            free_energy_reference = free_energy_reference,
                                            max_temperature = T_max,
                                            boltzmann_constant = k_B)

plt.plot(temperatures_temperature, free_energy_temperature_backward, label = "Backward, TEI")
plt.plot( mc.data_container.data["temperature"], mc.data_container.data["potential"], label = "Backward, TEI")

plt.legend()
plt.show()


