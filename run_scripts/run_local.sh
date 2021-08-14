cmd="python -m run_experiment -E cf --freq 500 -T 10500 -X 50 -Y 50 -W 0.1 --omega 1.0"
echo $cmd
$cmd

cmd="python -m run_experiment -E pf --freq 500 -T 10500 -X 50 -Y 50 -I 1.005 -O 1.0 --omega 1.0"
echo $cmd
$cmd

cmd="python -m run_experiment -E se -T 3000 --freq 150 -X 50 -Y 50 --eps 0.01 --rho 1.0 --omega 1.0 --mode d"
echo $cmd
$cmd

cmd="python -m run_experiment -E sv -T 3000 --freq 150 -X 50 -Y 50 --eps 0.01 --rho 1.0 --omega 1.0 --mode d"
echo $cmd
$cmd

cmd="python -m run_experiment -E se -T 3000 --freq 150 -X 50 -Y 50 --eps 0.01 --omega 1.0 --mode v"
echo $cmd
$cmd

cmd="python -m run_experiment -E sv -T 3000 --freq 150 -X 50 -Y 50 --eps 0.08 --omega 1.0 --mode v"
echo $cmd
$cmd

T=100000
F=5000
# Reynolds number: 250, 500, 750, 1000
for S in 75 150 225 300
do
    vel=0.1
    visc=0.03
    mpirun -n 9 python -m run_experiment -E sm -T $T --freq $F -X $S -Y $S --visc $visc -W $vel

    vel=0.2
    visc=0.06
    mpirun -n 9 python -m run_experiment -E sm -T $T --freq $F -X $S -Y $S --visc $visc -W $vel

    vel=0.3
    visc=0.09
    mpirun -n 9 python -m run_experiment -E sm -T $T --freq $F -X $S -Y $S --visc $visc -W $vel

    vel=0.4
    visc=0.12
    mpirun -n 9 python -m run_experiment -E sm -T $T --freq $F -X $S -Y $S --visc $visc -W $vel
done
