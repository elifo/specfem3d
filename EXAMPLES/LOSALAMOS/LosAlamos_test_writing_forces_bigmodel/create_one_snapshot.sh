#!/bin/bash

#
#  USAGE
#
#   ./create_one_snaphot <iteration> <DIRIN> <DISPLAY>
#
#   iteration is time step that you want to volume display
#
#

it=$(printf "%06d" $1)
DIRIN=$2
DISPLAY=$3
DIROUT=$4

echo "Iteration: $1"
echo "DIRIN: $2"
echo "DISPLAY: $3"
echo "DIROUT: $4"

# specfem bin directory
bin=./bin/


# available choices for DISPLAY :
#   velocity_X, velocity_Y, velocity_Z
#   curl_X, curl_Y, curl_Z
#   div_glob
#
#
#

# choose DATABASES_MPI DIRECTORY 2read

# choose output directory
mkdir -p $DIROUT

# choose resolution (low=0, high=1)
res=0

# --- DO NOT CHANGE------------------------------
declare -i NPROC
NPROC=`grep ^NPROC DATA/Par_file | grep -v -E '^[[:space:]]*#' | cut -d = -f 2`
NPROC="$NPROC-1"
#it=$(printf "%06d" $1)
$bin/xcombine_vol_data_vtk 0 $NPROC $DISPLAY"_it"$it $DIRIN $DIROUT $res

if [[ $? -ne 0 ]]; then exit 1; fi

