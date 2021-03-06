# Setup commands for BWUniCluster
module load devel/python/3.9.2_gnu_10.2
module load mpi/openmpi/4.1
pip install --user --upgrade mpi4py numpy matplotlib tqdm scipy

# Use only half of the CPUs for the correct performance measurement
# Note: 40 CPUs are available for each node

N_THREADS_LIST=(32 16 8 4 2 1)

for S in 100 300 1000
do
    for N_THREADS in ${N_THREADS_LIST[@]}
    do
        NODES=$((${N_THREADS}/20+1))
        echo "Use ${N_THREADS} threads on ${NODES} nodes."
        echo sbatch --nodes=$NODES --ntasks-per-node=40 run_server.sh -N $N_THREADS
        mpirun -n $N_THREADS python -m run_experiment -E sm -T 10000 -X $S -Y $S --visc 0.03 -W 0.1 --scaling True
        echo ""
    done
done
