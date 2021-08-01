cmd="python -m run_experiment -E cf -T 2400 -X 50 -Y 50 -W 0.5 --omega 0.35"
echo $cmd
$cmd

cmd="python -m run_experiment -E pf -T 5500 -X 50 -Y 50 -I 0.301 -O 0.300 --omega 0.70"
echo $cmd
$cmd

cmd="python -m run_experiment -E se -T 3000 -X 50 -Y 50 --eps 0.01 --rho 1.0 --omega 1.0 --mode d"
echo $cmd
$cmd

cmd="python -m run_experiment -E sv -T 3000 -X 50 -Y 50 --eps 0.01 --rho 1.0 --omega 1.0 --mode d"
echo $cmd
$cmd

cmd="python -m run_experiment -E se -T 3000 -X 50 -Y 50 --eps 0.01 --omega 1.0 --mode v"
echo $cmd
$cmd

cmd="python -m run_experiment -E sv -T 3000 -X 50 -Y 50 --eps 0.08 --omega 1.0 --mode v"
echo $cmd
$cmd

# The minimum possible viscosity
T=100000
vel=0.1
visc=0.03
mpirun -n 9 python -m run_experiment -E sm -T $T -X 300 -Y 300 --visc $visc -W $vel

vel=0.2
visc=0.06
mpirun -n 9 python -m run_experiment -E sm -T $T -X 300 -Y 300 --visc $visc -W $vel

vel=0.3
visc=0.09
mpirun -n 9 python -m run_experiment -E sm -T $T -X 300 -Y 300 --visc $visc -W $vel

vel=0.4
visc=0.12
mpirun -n 9 python -m run_experiment -E sm -T $T -X 300 -Y 300 --visc $visc -W $vel
