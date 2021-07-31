module load devel/python/3.9.2_gnu_10.2
module load mpi/openmpi/4.1
pip install --user --upgrade mpi4py numpy matplotlib tqdm scipy

NODES=3
echo sbatch --nodes=$NODES --ntasks-per-node=40 run_scripts/run_reynolds10000.sh -N 100
sbatch --nodes=$NODES --ntasks-per-node=40 run_scripts/run_reynolds10000.sh -N 100
