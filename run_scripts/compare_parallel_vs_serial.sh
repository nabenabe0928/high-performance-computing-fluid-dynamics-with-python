#############################################
# Those parameters do not change the result #
#############################################
# Wall velocity
vel=0.1
# Viscosity
visc=0.03
# The total time step of the simulation
T=100

#############################
# Domain related parameters #
#############################
# The lattice size in the x axis
X=300
# The lattice size in the y axis
Y=300
# The number of processes to compare
P=9

python -m run_experiment -E ss -T $T --freq $T -X $X -Y $Y --visc $visc -W $vel
mpirun -n $P python -m run_experiment -E sm -T $T --freq $T -X $X -Y $Y --visc $visc -W $vel
python -m src.utils.parallel_implementation_test -T $T -X $X -Y $Y --visc $visc -W $vel
