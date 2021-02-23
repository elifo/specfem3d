/*
!=====================================================================
!
!               S p e c f e m 3 D  V e r s i o n  3 . 0
!               ---------------------------------------
!
!    Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                             CNRS, France
!                      and Princeton University, USA
!                (there are currently many more authors!)
!                          (c) October 2017
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
!=====================================================================
*/


__global__ void kernel_3_cuda_device(realw* veloc,
                                     realw* accel,
                                     int size,
                                     realw deltatover2,
                                     realw* rmassx,
                                     realw* rmassy,
                                     realw* rmassz) {

  int id = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x;

  realw rx,ry,rz;
  realw ax,ay,az;
  // because of block and grid sizing problems, there is a small
  // amount of buffer at the end of the calculation
  if (id < size) {
    rx = rmassx[id];
    ry = rmassy[id];
    rz = rmassz[id];
    ax = accel[3*id  ]*rx;
    ay = accel[3*id+1]*ry;
    az = accel[3*id+2]*rz;

    accel[3*id]   = ax;
    accel[3*id+1] = ay;
    accel[3*id+2] = az;

    veloc[3*id]   += deltatover2*ax;
    veloc[3*id+1] += deltatover2*ay;
    veloc[3*id+2] += deltatover2*az;
  }
}

