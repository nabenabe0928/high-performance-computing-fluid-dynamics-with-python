#!/bin/bash -x
#SBATCH --time=03:00:00
#SBATCH -J MLUPS
#SBATCH --mem=32gb
#SBATCH --export=ALL
#SBATCH --partition=multiple

module load devel/python/3.9.2_gnu_10.2
module load mpi/openmpi/4.1

echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_JOB_CPUS_PER_NODE} cores each."
echo "Use ${N} threads."

mpirun -n $N python -m run_experiment -E sm -T 100000 -X 1500 -Y 1500 --visc 0.03 -W 0.2
