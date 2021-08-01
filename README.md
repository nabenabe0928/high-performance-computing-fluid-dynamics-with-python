# Lid-driven cavity using lattice boltzmann method
<p align="middle">
    <img src="/README_media/lid-driven-cavity-w0.1-visc0.03-small.gif" width="96%" />
</p>

# Preface
This repository is created for the final submission of 
the course `high performance computing fluid dynamics with python`
at the University of Freiburg 2021 Summer semester.

The following figures are the visualization of the velocity field of sliding lid at every 4000 time steps from 1000 to 45000 time steps.
<table>
    <tr>
        <td><img src="README_media/vel_flow001000.png" alt=""></td>
        <td><img src="README_media/vel_flow005000.png" alt=""></td>
        <td><img src="README_media/vel_flow009000.png" alt=""></td>
        <td><img src="README_media/vel_flow013000.png" alt=""></td>
    </tr>
    <tr>
        <td><img src="README_media/vel_flow017000.png" alt=""></td>
        <td><img src="README_media/vel_flow021000.png" alt=""></td>
        <td><img src="README_media/vel_flow025000.png" alt=""></td>
        <td><img src="README_media/vel_flow029000.png" alt=""></td>
    </tr>
    <tr>
        <td><img src="README_media/vel_flow033000.png" alt=""></td>
        <td><img src="README_media/vel_flow037000.png" alt=""></td>
        <td><img src="README_media/vel_flow041000.png" alt=""></td>
        <td><img src="README_media/vel_flow045000.png" alt=""></td>
    </tr>
</table>


# Setup

First, setup the environment using the following commands:
```
$ conda create -n hpc-fluid -c conda-forge python=3.6
$ conda activate hpc-fluid
$ pip install -r requirements.txt
```

# Reproduce the experiments

```
# The experiments except the scaling test
$ ./run_scripts/run_local.sh

# The scaling test (computations that require only 1 node)
$ ./run_scripts/run_on_cluster_manually.sh

# The scaling test (computations that require more than 1 node)
$ ./run_scripts/run_on_cluster.sh
```

Note that `./run_scripts/run_local.sh` includes the sliding lid simulation with the lattice shape of `300 x 300`
and you might want to avoid these computations.

# Structure

This repository is composed of three main parts:
1. Lattice Boltzmann method (`src/simulation_attributes/lattice_boltzmann_method.py`)
2. Boundary handling (`src/simulation_attributes/boundary_handling.py`)
3. Communication among threads (`src/utils/parallel_computation.py`)
The codes are securely maintained by the test codes in `test/`.

Furthermore, the visualizations and experiments are supported by `src/utils/visualization.py` and `src/experiments.py`.
All the codes are based on the codes created during the progress of milestones provided by lecturers.
However, since the final goal is to yield the results specified by the course, each milestone might not work because of the modifications that came later.
