python -m run_experiment -E cf -T 1600 -S 50 -W 0.5 --omega 0.35
python -m run_experiment -E pf -T 5000 -S 50 -I 0.301 -O 0.300 --omega 0.70
python -m run_experiment -E se -T 3000 -S 50 --eps 0.08 --rho 0.5 --omega 1.0 -S 50 --mode d
python -m run_experiment -E se -T 3000 -S 50 --eps 0.08 --omega 1.0 -S 50 --mode v
python -m run_experiment -E sv -T 3000 -S 50 --eps 0.08 --omega 1.0 -S 50 --mode v
python -m run_experiment -E sv -T 3000 -S 50 --eps 0.08 --rho 0.5 --omega 1.0 -S 50 --mode d

# The minimum possible viscosity
T=100000
vel=0.1
visc=0.03
mpirun -n 9 python -m run_experiment -E sm -T $T -S 300 --visc $visc -W $vel

vel=0.2
visc=0.06
mpirun -n 9 python -m run_experiment -E sm -T $T -S 300 --visc $visc -W $vel

vel=0.3
visc=0.09
mpirun -n 9 python -m run_experiment -E sm -T $T -S 300 --visc $visc -W $vel

vel=0.4
visc=0.12
mpirun -n 9 python -m run_experiment -E sm -T $T -S 300 --visc $visc -W $vel
