from helper_functions import *

from ase import Atoms
from icet import ClusterSpace, StructureContainer
from trainstation import Optimizer
from icet import ClusterExpansion
import os

#============================================================
#USER MODIFICATION SECTION START
#============================================================

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

AAC_directories = [
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

ATAT_format_lattice_path = "initial_data_files/lat.in"
output_file_name = "generated_data_files/cluster_expansion.ce"

cutoffs = [6, 4.5] #NOT SURE HOW TO HANDLE THIS YET

method_of_fitting = "ridge" #NOT SURE HOW TO HANDLE THIS YET

#============================================================
#USER MODIFICATION SECTION END
#============================================================

#============================================================
#READING AND PARSING ALL ATAT FORMAT FILES INTO ICET FORMAT
#DEFINING, CREATING SOME ICET SPECIFIC VARIABLES, INSTANCES
#SECTION START
#============================================================
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

print("\n================ CLUSTER SPACE ================\n")
print(cluster_space)

#============================================================
#READ, PARSE, DEFINE, CREATE SECTION END
#============================================================

#============================================================
#READ DATA FROM DFTS. FEED EVERYTHING TO CLUSTERS. SECTION START
#============================================================
training_data = []

final_directories = ["initial_data_files/" + i + "/" + j for i in GGI_directories for j in AAC_directories]

for directory in final_directories:

    print(directory)
    data = get_struct_and_energy(directory)

    training_data.append(
        (data[0], data[1]/216, directory)
    )

print("\n===============================================")
print(f"Found {len(training_data)} training structures")
print("===============================================\n")

structure_container = StructureContainer(
                                          cluster_space = cluster_space
                                        )

for structure, energy, directory in training_data:

    try:
        structure_container.add_structure(
                                           structure,
                                           properties={"energy": energy}
                                         )
    except Exception as e:
        print(f"\nSMTH WRONG WITH: {directory}")
        print(e)

#============================================================
#READ DATA FROM DFTS. FEED EVERYTHING TO CLUSTERS. SECTION END
#============================================================

#============================================================
#BRING EVERYTHING TO MATRIX FORMULA AJ = E, WHERE J ARE COEFFICIENTS OF INTEREST
#SECTION START
#============================================================

A, E = structure_container.get_fit_data(
                                         key="energy"
                                       )

#============================================================
#BRING EVERYTHING TO MATRIX FORMULA AJ = E, WHERE J ARE COEFFICIENTS OF INTEREST
#SECTION END
#============================================================

#============================================================
#FIT THE DATA TO OBTAIN COEFFS. CREATE CE AND SAVE IT.
#SECTION START
#============================================================

optimizer = Optimizer(
                       fit_data = (A, E),
                       fit_method = method_of_fitting
                     )

optimizer.train()


cluster_expansion = ClusterExpansion(
    cluster_space=cluster_space,
    parameters=optimizer.parameters,
)

print("\n================ CLUSTER EXPANSION ================\n")
print(cluster_expansion)

os.makedirs("generated_data_files", exist_ok=True)
cluster_expansion.write(output_file_name)

#============================================================
#FIT THE DATA TO OBTAIN COEFFS. CREATE CE AND SAVE IT.
#SECTION END
#============================================================