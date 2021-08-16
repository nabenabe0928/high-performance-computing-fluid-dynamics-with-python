# Setup commands for BWUniCluster
module load devel/python/3.9.2_gnu_10.2
module load mpi/openmpi/4.1
pip install --user --upgrade mpi4py numpy matplotlib tqdm scipy

# Use only half of the CPUs for the correct performance measurement
# Note: 40 CPUs are available for each node

N_THREADS_LIST=(64 128 256 512 1024 2048)
for S in 100 300 1000
do
    for N_THREADS in ${N_THREADS_LIST[@]}
    do
        NODES=$((${N_THREADS}/40+1))
        echo "Use ${N_THREADS} threads on ${NODES} nodes."
        echo sbatch --nodes=$NODES --ntasks-per-node=40 run_scripts/run_scaling_test.sh -N $N_THREADS -S $S
        sbatch --nodes=$NODES --ntasks-per-node=40 run_scripts/run_scaling_test.sh -N $N_THREADS -S $S
        echo ""
    done
done
