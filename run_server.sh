#!/bin/bash -x
#SBATCH --time=00:40:00
#SBATCH -J MLUPS
#SBATCH --mem=6gb
#SBATCH --ntasks-per-node=40
#SBATCH --export=ALL
#SBATCH --partition=multiple

module load devel/python/3.9.2_gnu_10.2
module load mpi/openmpi/4.1

echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_JOB_CPUS_PER_NODE} cores each."

# visc = 0.04 => the possible minimum viscosity
export EXP_NUM_THREADS=$((${SLURM_JOB_CPUS_PER_NODE}*${SLURM_JOB_NUM_NODES}))
time mpirun -n $EXP_NUM_THREADS python -m run_experiment -E sm -T 100000 -S 300 --visc 0.04 -W 0.1