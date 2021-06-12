pip install --user --upgrade mpi4py numpy

for NODES in `seq 1 50`; do
    echo sbatch --nodes=$NODES run_server.sh
    sbatch --nodes=$NODES run_server.sh
done