# I use this repo to debug big-mesh problem 
# when using dynamic rupture for Ridgecrest and
# Hetero paper (07/22)

# module load cuda/11.2 openmpi/4.1.1_cuda-11.2

./configure FC=gfortran MPIFC=mpif90  --with-cuda=cuda8

make clean 
make all

