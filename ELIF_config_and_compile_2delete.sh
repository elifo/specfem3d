# I use this repo to debug big-mesh problem 
# when using dynamic rupture for Ridgecrest and
# Hetero paper (07/22)

# module load cuda/11.2 openmpi/4.1.1_cuda-11.2

./configure FC=gfortran MPIFC=mpif90  --with-cuda=cuda8 MPI_INC=/usr/projects/hpcsoft/cos2/common/x86_64/intel-clusterstudio/2020.4.912/compilers_and_libraries_2020.4.304/linux/mpi/intel64/include/

make clean 
make all

