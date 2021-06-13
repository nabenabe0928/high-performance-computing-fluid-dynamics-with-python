module load devel/python/3.9.2_gnu_10.2
module load mpi/openmpi/4.1

pip install --user --upgrade mpi4py numpy matplotlib tqdm scipy

# 10
# 12
# 15
# 20
# 25
# 30
# Use only half of the CPUs for the correct performance measurement
# Note: 40 CPUs are available for each node

NTASKS=40
NS=(1 4 9 16 25 36 100 144 225 400 625 900)
for N in ${NS[@]}; do
    NODES=$((${N}/20+1))
    echo sbatch --nodes=$NODES --ntasks-per-node=$NTASKS run_server.sh -N $N
    sbatch --nodes=$NODES --ntasks-per-node=$NTASKS run_server.sh -N $N
done
