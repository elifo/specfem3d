# caltech cluster config
# module load cuda/11.2 openmpi/4.1.1_cuda-11.2
#./configure FC=gfortran MPIFC=mpif90  --with-cuda=cuda8

# modules to load
# cray-mpich/8.1.26  cuda/11.6

# Carene's for LANL Chicoma
./configure FC=ftn MPIFC=ftn CC=cc CXX=CC --with-cuda=cuda11 CUDA_INC="$CUDA_INCLUDES" CUDA_LIB="$CUDA_LIBS" MPI_INC=$MPICH_DIR/include

make clean 
make all

