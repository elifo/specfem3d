#!/bin/bash

echo "running example: `date`"
currentdir=`pwd`



cp DATA/Par_file OUTPUT_FILES/
cp DATA/CMTSOLUTION OUTPUT_FILES/
cp DATA/STATIONS OUTPUT_FILES/

# get the number of processors, ignoring comments in the Par_file
NPROC=`grep ^NPROC DATA/Par_file | grep -v -E '^[[:space:]]*#' | cut -d = -f 2`

BASEMPIDIR=`grep ^LOCAL_PATH DATA/Par_file | cut -d = -f 2 `


# links executables
mkdir -p bin
cd bin/
rm -f *
ln -s ../../../../bin/xspecfem3D
ln -s ../../../../bin/xcombine_vol_data_vtk
cd ../


# runs simulation
if [ "$NPROC" -eq 1 ]; then
  # This is a serial simulation
  echo
  echo "  running solver..."
  echo
  ./bin/xspecfem3D
else
  # This is a MPI simulation
  echo
  echo "  running solver on $NPROC processors..."
  echo
  mpirun -np $NPROC ./bin/xspecfem3D
fi
# checks exit code
if [[ $? -ne 0 ]]; then exit 1; fi



if [[ $? -ne 0 ]]; then exit 1; fi

echo
echo "see results in directory: OUTPUT_FILES/"
echo
echo "done"
echo `date`


