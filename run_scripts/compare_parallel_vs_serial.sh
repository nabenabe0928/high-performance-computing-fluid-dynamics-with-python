vel=0.1
visc=0.03
T=100
X=30
Y=30
P=9

python -m run_experiment -E ss -T $T --freq $T -X $X -Y $Y --visc $visc -W $vel
mpirun -n $P python -m run_experiment -E sm -T $T --freq $T -X $X -Y $Y --visc $visc -W $vel
python -m src.utils.parallel_implementation_test -T $T -X $X -Y $Y --visc $visc -W $vel
