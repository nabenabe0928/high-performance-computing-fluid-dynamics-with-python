#!/bin/bash -x
#SBATCH --time=00:40:00
#SBATCH -J MLUPS
#SBATCH --mem=200gb
#SBATCH --export=ALL
#SBATCH --partition=multiple

module load devel/python/3.9.2_gnu_10.2
module load mpi/openmpi/4.1

while getopts ":N:" o; do
    case "${o}" in
        N) N=${OPTARG};;
    esac
done

echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_JOB_CPUS_PER_NODE} cores each."
echo "Use ${N} threads."

# visc = 0.04 => the possible minimum viscosity
# Use only half of them for performance
time mpirun -n $N python -m run_experiment -E sm -T 100000 -S 300 --visc 0.04 -W 0.1 --scaling True
