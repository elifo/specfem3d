import numpy as np
import os
import array
import struct
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata as gd
#import pandas as pd
#from matplotlib.colors import LogNorm



def get_M0(Mw=7.0):
    return 10** (Mw* 1.5+ 9.1)
##

def get_Mw(M0=None):
    import numpy as np
    return (np.log10(M0)-9.1)/ 1.5
##

def interpolate_velocity_profile_for_specfem3d(points, Vs_values, my_points,\
                                          rep='./',nproc=1,is_1D_1layer=False,is_1D_layered=False,\
                                          x_chosen=None,y_chosen=None,z_chosen=None, \
                                          is_damage=False):

    from scipy.interpolate import griddata

    my_Vs, my_Vp, all_points = [], [], []
    for iproc in range(nproc):
        print ('iproc', iproc, nproc)
        print ('Reading read_specfem3d_database_coords ...')
        xproc, yproc, zproc = read_specfem3d_database_coords(iproc=iproc,rep=rep)
        # interpolate my points based on reference
        my_points_orig = np.array( [[_x,_y,_z] for _x,_y,_z in zip(xproc,yproc,zproc)] )# in meters
        if not is_1D_1layer and not is_1D_layered:
            print ('White et al. velocity model ...')
            my_points = my_points_orig
        elif is_1D_layered:
            print ('1D layered velocity model ...')        
            my_points = np.array( [[x_chosen*1e3,y_chosen*1e3,_z] for _x,_y,_z in zip(xproc,yproc,zproc)] )# in meters
        elif is_1D_1layer:
            print ('1D single-layer velocity model ...')                
            my_points = np.array( [[x_chosen*1e3,y_chosen*1e3,z_chosen*1e3] for _x,_y,_z in zip(xproc,yproc,zproc)] )# in meters        
        #
        print ('   interpolation ...')
        _my_Vs = griddata(points, Vs_values, my_points, method='nearest')
        # correction for LVFZs, only for Vs
        if is_damage:
            print ('   LVFZ correction ...')
            try:
                cdt1 = (yproc <= yend) & (yproc >= ybeg) & (xproc >= xbeg) & (xproc <= xend) & \
                      (zproc >= zbot) & (zproc <= ztop)
                print ('      _my_Vs[cdt1].shape: ', _my_Vs[cdt1].shape)
                print ('      min, max before: ', min(_my_Vs[cdt1]), max(_my_Vs[cdt1]))
                _my_Vs[cdt1] = _my_Vs[cdt1]* (1.0- reduction)
                print ('      min, max after: ', min(_my_Vs[cdt1]), max(_my_Vs[cdt1]))
            except:
                print ('      _my_Vs[cdt1].shape: ', _my_Vs[cdt1].shape)
                print ('      SKIPPED!')
                pass
        #
        my_Vs.append ( _my_Vs )        
        my_Vp.append ( griddata(points, Vp_values, my_points, method='nearest') )
        all_points.append(my_points_orig)
        print ('   done!'); print ()
    return all_points, my_Vs, my_Vp        
### 



def interpolate_from_reference_data(input_points, ref_dataframe=None,\
                                        data_type='mu', method='nearest'): 

    import pandas as pd
    from scipy.interpolate import griddata
    from scipy.interpolate import LinearNDInterpolator
    

    try:
        df_velocprofile = pd.read_pickle(ref_dataframe)
    except:
        import pickle5 as pickle
        with open(ref_dataframe, "rb") as fh:
            df_velocprofile = pickle.load(fh)   
    #
    # make a function for the reference model
    points = df_velocprofile[['x_rot', 'y_rot', 'depth']].to_numpy()
    points[:,2] *= -1 # depth to along-dip negative 
    points *= 1e3 # all coords in m
    reference_data = df_velocprofile[data_type]
    # interpolate
    data2return = griddata(points,reference_data,input_points,method=method) 
    return data2return
##


def get_rotd50_optim_testing(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
        percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):

    from intensity_measures import get_response_spectrum, rotate_horizontal

    # Get the time-series corresponding to the SDOF
    sax, _, x_a, _, _ = get_response_spectrum(acceleration_x,
                                              time_step_x,
                                              periods, damping,
                                              units, method)
    say, _, y_a, _, _ = get_response_spectrum(acceleration_y,
                                              time_step_y,
                                              periods, damping,
                                              units, method)

    angles = np.arange(0., 90., 1.)
    max_a_theta = np.zeros([len(angles), len(periods)], dtype=float)
    max_a_theta[0, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) *
                                np.max(np.fabs(y_a), axis=0))
  
    iloc, theta = 0, angles[0]
    max_a_theta[iloc, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) *
                                   np.max(np.fabs(y_a), axis=0))    

    for iloc, theta in enumerate(angles[1:]):
        rot_x, rot_y = rotate_horizontal(x_a, y_a, theta)
        max_a_theta[iloc, :] = np.sqrt(np.max(np.fabs(rot_x), axis=0) *
                                       np.max(np.fabs(rot_y), axis=0))
###

        

    gmrotd = np.percentile(max_a_theta, percentile, axis=0)
    return {"angles": angles,
            "periods": periods,
            "GMRotDpp": gmrotd,
            "GeoMeanPerAngle": max_a_theta}
###

def set_style(whitegrid=False, scale=0.85):
      sns.set(font_scale = scale)
      sns.set_style('white')
      if whitegrid: sns.set_style('whitegrid')
      sns.set_context('talk', font_scale=scale)
###


def get_FFT(dt=None, data=None):    
    from scipy.fft import fft
    
    # Spectrum    
    df=1.0/ dt
    N = len(data)
    f = np.linspace(0.0, 1.0/(2.0*dt), N//2)
    spec = abs(fft(data))* dt    
    return f[1:], spec[:int(N/2)-1]
###


def prepare_station_file(xrange=(),yrange=(),zrange=(),n_xyz=(),USE_SOURCES_RECEIVERS_Z=True,\
        sta='EL',net='IF',file_stat='w'):
    ''' prepares the station file STATIONS to get seismogram outputs'''
    import numpy as np
    
    try:
        nx, ny, nz = n_xyz[0], n_xyz[1], n_xyz[2]
        if USE_SOURCES_RECEIVERS_Z:
            print('Using z-coordinates, do not forget to set USE_SOURCES_RECEIVERS_Z=T in Par_file!')
            ('VERIFY file format: which column is z-coord?')
        else:
            print('Using burial depth, do not forget to set USE_SOURCES_RECEIVERS_Z=F in Par_file!')
            print ('z-coordinates in this file must be positive!')   
        ista = 0
        with open('STATIONS', file_stat) as f:
            for _x in np.linspace(xrange[0], xrange[1], nx):
                for _y in np.linspace(yrange[0], yrange[1], ny):
                    for _z in np.linspace(zrange[0], zrange[1], nz):
                        name = sta+'%06d' % (ista)
                        f.write('%s \t %s \t  %.1f \t %.1f \t  %s \t %s \n' \
                                    % (name, net, _y, _x, '0.0', _z) )                      
                        ista += 1
        print ('Total number of stations: ', ista)
        print ('STATIONS file ready!')
        print ('*')
    except:
        print ('Provide xrange=(xmin,xmax),yrange=(ymin,ymax),zrange=(zmin,zmax),n_xyz=(nx,ny,nz)')
        print ()
# ##

def get_JA19_spectrum(Mw=None, f=None):
    ''' Ji and Archuleta (2020) spectrum with double corner
    frequency. Here using only Mw>5.3 condition.'''
    print ('get_JA19_spectrum: assumes Mw > 5.3 !')
    fc1_JA19_2S = 2.375- 0.585* Mw # for M>5.3
    fc1_JA19_2S = 10.0** fc1_JA19_2S
    fc2_JA19_2S = 3.250- 0.5* Mw
    fc2_JA19_2S = 10.0** fc2_JA19_2S    
    coeff = (1.0+ (f/ fc1_JA19_2S)** 4.0)** 0.25
    coeff *= (1.0+ (f/ fc2_JA19_2S)** 4.0)** 0.25
    spec_JA19_2S = 1.0/ coeff        
    return fc1_JA19_2S, fc2_JA19_2S, spec_JA19_2S
##

# taken from Huihui's (02/2019)
def _read_binary_fault_file(filename, single=False, ndata=14):

      # Precision
      length = 8 
      if single: length = 4

      if os.path.exists(filename):  

            BinRead = []
            data = type("",(),{})()

            with open(filename, 'rb') as fid:
                  for ii in range(ndata):
                        read_buf = fid.read(4)
                        number = struct.unpack('1i',read_buf)[0]
                        N = int(number/length)
                        read_buf = fid.read(number)
                        read_d = struct.unpack(str(N)+'f',read_buf)
                        read_d = np.array(read_d)
                        BinRead.append(read_d)
                        read_buf = fid.read(4)
                        number = struct.unpack('1i',read_buf)[0]
            fid.close()

            # assign the values to parameters
            data.X  = BinRead[0]/ 1.e3   # in km
            data.Y  = BinRead[1]/ 1.e3   # in km
            data.Z  = BinRead[2]/ 1.e3   # in km
            data.Dx = BinRead[3]
            data.Dz = BinRead[4]
            data.Vx = BinRead[5]
            data.Vz = BinRead[6]
            data.Tx = BinRead[7]      # in MPa
            data.Ty = BinRead[8]
            data.Tz = BinRead[9]     # in MPa
            data.S  = BinRead[10]
            data.Sg = BinRead[11]     # in MPa
            data.Trup = BinRead[12]
            data.Tproz = BinRead[13]
      else:
            print ('No such file or directory!')
            return
      return data
###


# Konno-Ohmachi smoothening
def ko(y,dx,bexp):
    
    from math import pi, log10, sin

    nx      = len(y)
    fratio  = 10.0**(2.5/bexp)
    ylis    = np.zeros(nx) #np.arange( nx )
    ylis[0] = y[0]

    for ix in np.arange( 1,nx ):
        fc  = float(ix)*dx
        fc1 = fc/fratio
        fc2 = fc*fratio
        ix1 = int(fc1/dx)
        ix2 = int(fc2/dx) + 1
        if ix1 <= 0:  ix1 = 1
        if ix2 >= nx: ix2 = nx
        a1 = 0.0
        a2 = 0.0
        for j in np.arange( ix1,ix2 ):
            if j != ix:
                c1 = bexp* np.log10(float(j)* dx/ fc)
                c1 = (sin(c1)/c1)**4
                a2 = a2+c1
                a1 = a1+c1*y[j]
            else:
                a2 = a2+1.0
                a1 = a1+y[ix]
        ylis[ix] = a1 / a2

    for ix in np.arange( nx ):
        y[ix] = ylis[ix]
    return y
###


def compute_STF(t, all_SR, all_mu, dx=0.1e3, dz=0.1e3, pad=False):
    ''' computes STF from slip rates of fault stations. NOT fault grid.
    See also compute_STF_from_faultgrid in lib_ridgecrest.py'''
    from scipy import fft
    N = len(all_SR[0])
    STF = np.zeros(N)
    elemsize = dx*dz
    # STF
    for _mu, SR in zip(all_mu, all_SR):
        STF += SR* elemsize* _mu
    dt = t[1]- t[0]
    if pad:
        STF = np.pad(STF, pad_width=(0,len(STF)), mode='minimum')        
        t = np.arange(len(STF))* dt
    # Spectrum    
    df=1.0/ dt
    N = len(t)
    f = np.linspace(0.0, 1.0/(2.0*dt), N//2)
    spec = abs(fft(STF))* dt    
    specD=spec[:int(N/2)-1]
    specA=(2*np.pi*f[1:])**2*specD
    return t, STF, f[1:], specD, specA
###


def get_seismo_fast_1component(self, sta, filtering=False, fmin=None, fmax=1.0,
                        is_binary=False,is_acceleration=False,suffix='H',compo='XX.semv'):

  from obspy.core import Trace

      
  fname = self.directory+'/'+sta+'.'+suffix+compo

  if is_binary:
      with open(fname,'rb') as f: Vx = np.fromfile(f, dtype=np.float32)
      t = np.arange(len(Vx))* self.dt
  else:
      t = np.genfromtxt(fname, usecols=0)
      Vx = np.genfromtxt(fname, usecols=1)
  ###    
  tr_x = Trace()
  tr_x.data = Vx
  tr_x.stats.delta = t[1]-t[0]
  tr_x.stats.starttime = t[0]
  if filtering: tr_x.filter(type='lowpass', freq=fmax, corners=2, zerophase=True)
  if fmin != None: tr_x.filter(type='highpass', freq=fmin, corners=2, zerophase=True)
  if is_acceleration: tr_x.differentiate() 

  return tr_x, max(abs(Vx)), max(abs(tr_x.data))
###


def get_scec_velocity_profile(zcoords):
    # depth coords in km
    zandrews = [abs(depth)/1e3+ 0.0073215 for depth in zcoords] # in km
#     print ('Min and max depth of zandrews: ', min(zandrews), max(zandrews))
    vs0 = np.zeros(len(zandrews))
    vs_not_corrected = np.zeros(len(zandrews))
    zandrews = np.array(zandrews)
    
    # ATTENTION TO THE ORDER
    cdt = (zandrews >= 8.0)
    vs0[cdt] = 2.927* 8.0**0.086

    cdt = (zandrews < 8.0)
    vs0[cdt] = 2.927* zandrews[cdt]**0.086

    cdt = (zandrews < 4.0)
    vs0[cdt] = 2.505* zandrews[cdt]**0.199
    
    cdt = (zandrews < 0.19)
    vs0[cdt] = 3.542* zandrews[cdt]**0.407    
    
    cdt = (zandrews < 0.03)
    vs0[cdt] = 2.206* zandrews[cdt]**0.272    
    
    vs_not_corrected[:] = vs0[:]*1e3              
    vp0_not_corrected = [ max(1.4+ 1.14*vs, 1.68*vs) for vs in vs_not_corrected]
    vp0_not_corrected = np.array(vp0_not_corrected)   
        
    # clamp surface Vs_min to 760 m/s and update Vp and rho
    # assuming deeper points have already > 760 !
#     vs0 [zcoords == 0.0] = 760.0e-3
    vs0 [vs0 == min(vs0)] = 760.0e-3
    vp0 = [ max(1.4+ 1.14*vs, 1.68*vs) for vs in vs0]
    vp0 = np.array(vp0)    
    rho0 = [2.4405+ 0.10271*vs  for vs in vs0]
    rho0 = np.array(rho0)          
    vp0 *= 1e3
    vs0 *= 1e3
    rho0 *= 1e3

    return vp0, vs0, rho0, rho0*vs0*vs0
###


### Scaling law of Wells& Coppersmith (94)
def get_WC94_scaling(Mw=7.0):
   ''' Earthquake scaling law based on the WC94
       regression formulas of strike-slip events'''

   WC94 = {}

   # Surface rupture length
   a = [5.16-0.13, 5.16, 5.16+0.13]
   b = [1.12-0.08, 1.12, 1.12+0.08]
   WC94['SRL'] = [10.0** ( (Mw-_a)/_b )  for _a, _b in zip(a,b) ]

   # Subsurface rupture length
   a = [4.33-0.06, 4.33, 4.33+0.06]
   b = [1.49-0.05, 1.49, 1.49+0.05]
   WC94['RLD'] = [10.0** ( (Mw-_a)/_b )  for _a, _b in zip(a,b) ]

   # Rupture width
   a = [3.80-0.17, 3.80, 3.80+0.17]
   b = [2.59-0.18, 2.59, 2.59+0.18]
   WC94['RW'] = [10.0** ( (Mw-_a)/_b )  for _a, _b in zip(a,b) ]

   # Rupture area
   a = [3.98-0.07, 3.98, 3.98+0.07]
   b = [1.02-0.03, 1.02, 1.02+0.03]
   WC94['RA'] = [10.0** ( (Mw-_a)/_b )  for _a, _b in zip(a,b) ]

   # Max surface displacement
   a = [6.81-0.05, 6.81, 6.81+0.05]
   b = [0.78-0.06, 0.78, 0.78+0.06]
   WC94['MD'] = [10.0** ( (Mw-_a)/_b )  for _a, _b in zip(a,b) ]

   # Ave surface displacement
   a = [7.04-0.05, 7.04, 7.04+0.05]
   b = [0.89-0.09, 0.89, 0.89+0.09]
   WC94['AD'] = [10.0** ( (Mw-_a)/_b )  for _a, _b in zip(a,b) ]

   return WC94
###

### CLASS ###
class specfem3d (object):

      def __init__(self, directory='', n_fault=1, n_iter=1, itd=1, is_single=False,is_snapshot=False, info=True):
            self.directory = directory+ '/'
            self.fault = {}
            if info: print ('Directory: ', self.directory); print ();
            self.itd = itd
            if is_snapshot: self.__read_snapshots(n_fault=n_fault, n_iter=n_iter, itd=itd, single=is_single,_info=info)
            ###

      def __read_snapshots(self, n_fault=1, n_iter=1, itd=1, single=False, _info=True):

            if _info: print ('Number of snapshots: ', n_iter)
            if _info: print ('Reading...')
            # For each snapshot:
            snap=0; all_data=[];
            for j in range(n_iter) :
                  u = str(int(snap))
                  BinFile = self.directory+ 'Snapshot'+u+'_F'+str(int(n_fault))+'.bin'
                  if _info: print ('Binary file: ', BinFile)
                  data = _read_binary_fault_file (BinFile, single=single, ndata=14)
                  all_data.append(data)
                  snap+=itd
            ###
            self.fault['x'] = all_data[0].X
            self.fault['y'] = all_data[0].Y
            self.fault['z'] = all_data[0].Z
            self.fault['Dx'] = [data.Dx for data in all_data]
            self.fault['Dz'] = [data.Dz for data in all_data]
            self.fault['Vx'] = [data.Vx for data in all_data]
            self.fault['Vz'] = [data.Vz for data in all_data]
            self.fault['Tx'] = [data.Tx for data in all_data]
            self.fault['Ty'] = [data.Ty for data in all_data]
            self.fault['Tz'] = [data.Tz for data in all_data]
            self.fault['S']  = [data.S for data in all_data]
            self.fault['Sg'] = [data.Sg for data in all_data]
            self.fault['Trup'] = [data.Trup for data in all_data]
            self.fault['Tproz'] = [data.Tproz for data in all_data]
            print('***')
         ###


      def plot_final_slip(self, vertical_fault=True, cmap='magma',tmax=0.0, Nx=1000, Nz=1000,
                                  Xcut=None,Zcut=None,ext=[],info=True,Ncmap=6,plot=True,\
                                  **kwargs):

          if not vertical_fault: print('Modify the script for non-vertical faults!!!')

          if tmax ==0.0: nsnaps = len(self.fault['Dx'])
          else: nsnaps = int(tmax/self.dt/self.itd)

          if info: print ('Total number of used snapshots: ', nsnaps)
          Dx_all = self.fault['Dx'][:nsnaps]
          Dz_all = self.fault['Dz'][:nsnaps]

          Xall = self.fault['x'] 
          Y = self.fault['y']  
          Zall = self.fault['z']  

          if Zcut==None: Zcut = min(Zall)
          if Xcut==None: 
              Xcutmin, Xcutmax = min(Xall), max(Xall)
          else: 
              Xcutmin, Xcutmax = Xcut[0], Xcut[1]

          cdt = (Y == 0.0) & (Zall >= Zcut) & (Xall >= Xcutmin) & (Xall <= Xcutmax)
          X = Xall [cdt]
          Z = Zall [cdt]

          if ext==[]: ext = [min(X), max(X), min(Z), max(Z)]
          xi, zi = np.meshgrid(np.linspace(ext[0],ext[1], Nx), np.linspace(ext[2],ext[3],Nz))

          D1 = Dx_all[len(Dx_all)-1]
          D2 = Dz_all[len(Dz_all)-1]
          D_net = [(_D1**2+ _D2**2)**0.5  for _D1,_D2 in zip(D1,D2)  ]
          D_net = np.array(D_net)

          slip_direction = kwargs.pop('slip_direction', 'along_strike') 
          if slip_direction == 'along_dip':
              if info: print ('Using slip along dip! ')
              D = D2
              vmin=min(map(min, Dz_all)); vmax=max(map(max, Dz_all))
              vmax = max(abs(vmin), vmax); vmin = -vmax;
          elif slip_direction == 'both':
              if info: print ('Using composite slip (both components))')
              D = D_net  
              vmin=min(map(min, Dx_all)); vmax=max(map(max, Dx_all))
              vmax = max(abs(vmin), vmax); vmin=0.0  
          elif slip_direction == 'along_strike':
              if info: print ('Using slip along strike! ')              
              D = D1
              vmin=min(map(min, Dx_all)); vmax=max(map(max, Dx_all))
              vmax = max(abs(vmin), vmax); vmin=0.0      
          #

          vmax = kwargs.pop('vmax', vmax) 

          if info:
            print ('*') 
            print ('ON FAULT PLANE ::') 
            print ('*')             
            print ('Along strike:')
            print ('Max slip (m): ', max(D1) )
            print ('Ave slip (m): ', np.average(D1) )            
            print ('*')             
            print ('Along dip:')
            print ('Max slip (m): ', max(D2) )
            print ('Ave slip (m): ', np.average(D2) )
            print ('*')             
            print ('Both components:')
            print ('Max slip (m): ', max(D_net) )
            print ('Ave slip (m): ', np.average(D_net) )
            print ('*')             

            print ('*')             
            print ('ON SURFACE ::') 
            print ('*')             
            print ('Along strike:')
            D_surf = D1 [ (abs(Zall)== min(abs(Zall))) ] 
            print ('Max slip (m): ', max(D_surf) )
            print ('Ave slip (m): ', np.average(D_surf) )            
            print ('*')             
            print ('Along dip:')
            D_surf = D2 [ (abs(Zall)== min(abs(Zall))) ] 
            print ('Max slip (m): ', max(D_surf) )
            print ('Ave slip (m): ', np.average(D_surf) )   
            print ('*')             
            print ('Both components:')
            D_surf = D_net [ (abs(Zall)== min(abs(Zall))) ] 

          if not info:
            D_surf = D_net [ (abs(Zall)== min(abs(Zall))) ] 
            cdt2 = (abs(Zall)== min(abs(Zall)))
            x_surf = Xall[cdt2]
            print ('Max slip (m): ', max(D_surf) )
            print ('Ave slip (m): ', np.average(D_surf) )   
            print ('*')    
          ##
          if plot:
            plt.close('all')
            fig = plt.figure(figsize=(6,4))
            cmap = plt.cm.get_cmap(cmap, Ncmap)
            ### strike slip
            D = D [ cdt]                                
            y = gd( (X,Z), D, (xi,zi), method=kwargs.pop('interpolation', 'linear'), 
                                      fill_value=kwargs.pop('fill_value', 0.0))
            y = np.flipud(y)  
            ax = fig.add_subplot(111)
            ax.set_title('Max slip on fault plane (m) = '+ '%.2f' % (max(D))  )
            ax.set_xlabel('Along strike (km)', fontsize=15)
            ax.set_ylabel('Along dip (km)', fontsize=15)    
            im = ax.imshow(y, extent=ext, \
                                    vmin=vmin, vmax=vmax, cmap=cmap)
            try: ax.plot(self.x_hypo, self.z_hypo, c='k', marker='*', alpha=0.5)
            except: pass
            c = plt.colorbar(im, fraction=0.046, pad=0.05, shrink=0.5, label='Strike slip (m)')
            c.mappable.set_clim(vmin, vmax)
            plt.tight_layout()
            plt.show()

          return x_surf, D_surf
      ###

      def plot_surface_slip(self,D_surf,x_surf,percent=5.0):
        ''' plots surface slip variation'''
        try:
          cdt = [D_surf >= max(D_surf)*0.01*percent][0]
          print ('min-max surface locations: ', min(x_surf[cdt]), max(x_surf[cdt]))
          surfacelen = max(x_surf[cdt])- min(x_surf[cdt])
          print ('Surface length: ', surfacelen)
          print ('Ave. slip here: ', np.average(D_surf[cdt]))
          print ('percentage of clipping: ', percent)
          #
          plt.figure(figsize=(6,3))
          plt.subplot(111)
          plt.scatter(x_surf[cdt], D_surf[cdt], c='tomato', s=50)
          plt.scatter(x_surf, D_surf, c='gray', alpha=0.1)
          plt.scatter(self.x_hypo, 0.1, marker='*', c='gold', s=100)
          plt.title('Surface slip (m) vs Fault strike (km)')
          plt.axvline(x=self.x_hypo, c='gray', alpha=0.5, linestyle=':')
          plt.grid()
        except:
          print ('prerequisite: SEM.plot_final_slip()')
          print ('input format: D_surf,x_surf,percent=5.0')
          print ('*')
      ##

      def plot_snapshots(self, vertical_fault=True, n_contour=5, dt_snap=1.0, contour=False, 
                                    cmap='magma',_asp=3, data_type='D',**kwargs):

            if not vertical_fault: print('Modify the script for non-vertical faults!!!')

            Dx_all = self.fault[data_type+'x']
            Dz_all = self.fault[data_type+'z']

            X = self.fault['x'] 
            Y = self.fault['y']  
            Z = self.fault['z']  

            X = X [ (Y == 0.0) ]
            ext = [min(X), max(X), min(Z), max(Z)]
            xi, zi = np.meshgrid(np.linspace(ext[0],ext[1], 1e3), np.linspace(ext[2],ext[3],1e3))

            vmin=min(map(min, Dx_all)); vmax=max(map(max, Dx_all))
            vzmin=min(map(min, Dz_all)); vzmax=max(map(max, Dz_all))
            print ('Min and max of overall strike slip (m): ', '%.0e %.0e' % (vmin, vmax) )
            print ('Min and max of overall dip slip (m): ', '%.0e %.0e' % (vzmin, vzmax))

            vmax = max(abs(vmin), vmax); #vmin = -vmax        
            vzmax = max(abs(vzmin), vzmax); vzmin = -vzmax

            plt.close('all')
            for n, (D, Dz)in  enumerate( zip(Dx_all, Dz_all)):
                  print ('*') 
                  print ('t (s) = ', n*dt_snap)
                  print ('STRIKE SLIP - Min and max: ',  '%.0e %.0e' %  (min(D), max(D)) )
                  print ('DIP SLIP    - Min and max: ', '%.0e %.0e' %  (min(Dz), max(Dz)) )
                  ###

                  fig = plt.figure(figsize=(8,3.5)); set_style('whitegrid', scale=0.8)
                  
                  tit = 'Snapshot at time (s): '+ str(n*dt_snap)
                  plt.suptitle(tit)

                  ### strike slip
                  D = D [ (Y== 0.0) ]                                
                  y = gd( (X,Z), D, (xi,zi), method='linear')
                  y = np.flipud(y)  

                  ax = fig.add_subplot(121, aspect=_asp)
                  ax.set_title('Strike slip (m)')
                  ax.set_xlabel('Along strike (km)')
                  ax.set_ylabel('Along dip (km)')    
                  im = ax.imshow(y, extent=[min(X), max(X), min(Z), max(Z)], \
                                    vmin=vmin, vmax=vmax, cmap=cmap)

                  levels = np.linspace(0.0, vmax, num=n_contour)
                  if contour: ax.contour(y, levels, colors='white', alpha=0.5, extent=[min(X), max(X), min(Z), max(Z)],\
                               origin='upper')

                  c = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1, shrink=1, pad=0.25, format='%.0e')                
                  c.mappable.set_clim(vmin, vmax)


                  ### dip slip
                  Dz = Dz [ (Y == 0.0) ]                                
                  y = gd( (X,Z), Dz, (xi,zi), method='linear')
                  y = np.flipud(y)

                  ax = fig.add_subplot(122)
                  ax.set_title('Dip slip (m)')
                  ax.set_xlabel('Along strike (km)')
                  ax.set_ylabel('Along dip (km)')    

                  im = ax.imshow(y, extent=[min(X), max(X), min(Z), max(Z)], \
                                    vmin=vzmin, vmax=vzmax, cmap=cmap)

                  levels = np.linspace(0.0, vzmax, num=n_contour)
                  if contour: ax.contour(y, levels, colors='white', alpha=0.5, extent=[min(X), max(X), min(Z), max(Z)],\
                               origin='upper')
                  # c = plt.colorbar(im, fraction=0.046, pad=0.1,shrink=0.25)
                  c = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1, shrink=1, pad=0.25, format='%.0e')                
                  c.mappable.set_clim(vzmin, vzmax)

                  plt.tight_layout()
                  plt.show()
               ###
      ###


      def plot_snapshots_in1figure(self, vertical_fault=True, n_contour=5, contour=False, 
                         cmap='magma',_asp=3,jump=0, nrows=3, ncols=2, figsize=(8,8), ext=[], \
                                        dt_snap=1.0,tmax=24.0, ylim=-20.0, _vmax=-1.0, \
                                        save=False, hypo=False, info=False, plot_cbar=False,
                                        figname='',Ngrid=1000,xticks=[],yticks=[],\
                                        data_type='D'):

          if not vertical_fault: print('Modify the script for non-vertical faults!!!')

          if jump==0: jump = int(dt_snap)
          print ('jump', jump)
          print ('dt_snap ', dt_snap)
          
          if (nrows*ncols < int(tmax/dt_snap)): print('No space for all snapshots!')    
          
          # plot time interval is jump 
          Dx_all = self.fault[data_type+'x'][::jump]
          Dz_all = self.fault[data_type+'z'][::jump]

          X = self.fault['x'] 
          Y = self.fault['y']  
          Z = self.fault['z']  

          X = X [ (Y == 0.0) ] # assuming fault is located at y=0
          if ext==[]: ext = [min(X), max(X), min(Z), max(Z)]
          
          print ('Snapshot extent:', ext)
          xi, zi = np.meshgrid(np.linspace(ext[0],ext[1], Ngrid), np.linspace(ext[2],ext[3],Ngrid))

          vmin=min(map(min, Dx_all)); vmax=max(map(max, Dx_all))
          vzmin=min(map(min, Dz_all)); vzmax=max(map(max, Dz_all))
          if info: print ('Min and max of overall strike slip (m): ', '%.0e %.0e' % (vmin, vmax) )
          if info: print ('Min and max of overall dip slip (m): ', '%.0e %.0e' % (vzmin, vzmax))

          vmax = max(abs(vmin), vmax); #vmin = -vmax        
          vzmax = max(abs(vzmin), vzmax); vzmin = -vzmax

          if _vmax > 0.0: vmax=_vmax 
          print ('vmax here: ', vmax)

          plt.close('all')
          jj=0
          fig = plt.figure(figsize=figsize); set_style(whitegrid=False, scale=0.8)
          for n, (D, Dz)in  enumerate( zip(Dx_all[1:], Dz_all[1:])):
              t_sim = (n+1)* dt_snap
              
              if info: print ('*') 
              if info: print ('t (s) = ', t_sim)
              if info: print ('STRIKE SLIP - Min and max: ',  '%.0e %.0e' %  (min(D), max(D)) )
              ###

              # Subplot order: vertical arrangement
              habitus=[]
              for iax, icol in enumerate(np.arange(ncols)):
                   habitus += list(np.arange(iax+1, (nrows-1)*ncols+iax+1+1, ncols))
              habitus = habitus[jj]


              ### strike slip
              D = D [ (Y== 0.0) ]                                
              y = gd( (X,Z), D, (xi,zi), method='linear')
              y = np.flipud(y)  

              ###
              ax = fig.add_subplot(nrows,ncols,habitus, aspect=_asp)
              tit = '%.0f' % (t_sim)+ '  s'
              ax.set_title(tit)
              ax.set_ylim(ylim, 0.0)
              ax.set_yticks(yticks); ax.set_xticks(xticks)
              plt.yticks(fontsize=9);  plt.xticks(fontsize=9)              
              
              # hypocenter
              if hypo: ax.plot(self.x_hypo, self.z_hypo, c='k', marker='*', alpha=0.5)
              im = ax.imshow(y, extent=ext, vmin=vmin, vmax=vmax, cmap=cmap)

              levels = np.linspace(0.0, vmax, num=n_contour)
              if jj>=9: contour = True # to only show rupture extent in the last snap (change limit jj if necess.)
              if contour: ax.contour(y, levels, colors='white', alpha=1, extent=ext,\
                               origin='upper', linewidths=[1])

              print ('jj, t_sim', jj, t_sim)

              jj += 1                            
              if t_sim >= tmax: break               
          ### 

          plt.tight_layout()
          if save: fig.savefig(figname, dpi=300)
          plt.show()
      ###

      def read_all_fault_stations(self, search_tip='/x*y_*dat',is_pickled=False,rep_store='./STORE/'):
          ''' reads and pickles fault station outputs. Returns (x,z) coordinates, time and STF.'''
          import glob
          import pandas as pd
          #
          if not is_pickled:
            # read and pickle
            filenames = glob.glob(self.directory+search_tip)
            all_x, all_z, all_SR, all_slip = [],[],[],[]
            all_tau_xy, all_tau_yz, all_tau_yy = [],[],[]
            # get all filenames of fault stations
            print ('Reading fault station outputs...')
            for fname in filenames:
              # coordinates
              _beg = fname.find('x_')+2 
              _end = fname.find('_y')
              _xsta = float(fname[_beg:_end])

              _beg = fname.find('y_')+2 
              _end = fname.find('.dat')
              _zsta = float(fname[_beg:_end])
              # x: along strike; z: along dip      
              data = pd.read_csv(fname, names=('t','Dx','Vx','tau_xy', 'Dz', 'Vz', 'tau_yz', 'tau_yy' ), \
                            delim_whitespace=True, header=20)  
              # slip-rate fnc
              _t = data['t'].values
              _Vx = data['Vx'].values
              _Vz = data['Vz'].values
              _SR = (_Vx**2+ _Vz**2)** 0.5
              _Dx = data['Dx'].values
              _Dz = data['Dz'].values   
              _slip = (_Dx**2+ _Dz**2)** 0.5       
              _tau_xy = data['tau_xy'].values
              _tau_yz = data['tau_yz'].values
              _tau_yy = data['tau_yy'].values
              all_x.append(_xsta); all_z.append(_zsta)
              all_SR.append(_SR); all_slip.append(_slip) 
              all_tau_xy.append(_tau_xy)
              all_tau_yz.append(_tau_yz)
              all_tau_yy.append(_tau_yy)              
            # pickle     
            print ('Pickling ...')
            try: os.makedirs(rep_store)
            except: pass
            np.save(rep_store+'/all_x.npy', all_x)
            np.save(rep_store+'/all_z.npy', all_z)
            np.save(rep_store+'/all_t.npy', _t)        
            np.save(rep_store+'/all_SR.npy', all_SR)   
            np.save(rep_store+'/all_slip.npy', all_slip)   
            np.save(rep_store+'/all_tau_xy.npy', all_tau_xy)   
            np.save(rep_store+'/all_tau_yz.npy', all_tau_yz)   
            np.save(rep_store+'/all_tau_yy.npy', all_tau_yy)   
          else:  
            print ('Opening pickle box ...')  
            all_x = np.load(rep_store+'/all_x.npy')
            all_z = np.load(rep_store+'/all_z.npy')  
            _t = np.load(rep_store+'/all_t.npy')        
            all_SR = np.load(rep_store+'/all_SR.npy')  
            all_slip = np.load(rep_store+'/all_slip.npy')  
            all_tau_xy = np.load(rep_store+'/all_tau_xy.npy')  
            all_tau_yz = np.load(rep_store+'/all_tau_yz.npy')  
            all_tau_yy = np.load(rep_store+'/all_tau_yy.npy')    
          #
          print ('Done!')
          self.faultsta_x = all_x
          self.faultsta_z = all_z
          self.fault_stf_t = _t
          self.fault_stf_SR = all_SR
          self.faultsta_slip = all_slip
          self.faultsta_tau_xy = all_tau_xy
          self.faultsta_tau_yz = all_tau_yz
          self.faultsta_tau_yy = all_tau_yy
      ##

      # def plot_STF_and_spectrum(self,plot_source=True,jump_ko=10,\
      #                   pow_ko=20.0,fmax=10.0, \
      #                   is_smooth=True,is_padding=False,\
      #                   is_pickled=False,rep_store='./STORE/'):
      #     ''' computes STF and its spectrum. compute Mw. plots time and frequncy functions 
      #     with Brune\'s and Ji and Archuleta's spectra.'''
      #     if not is_pickled:
      #       try:
      #         import numpy.ma as ma
      #         from scipy import fft
      #         xx, zz = np.array(self.faultsta_x), np.array(self.faultsta_z)  
      #         t = self.fault_stf_t
      #         fault_mu = self.faultsta_mu        
      #         dx, dz = max(np.diff( np.sort(xx)) )* 1e3, max(np.diff( np.sort(zz)) )* 1e3
      #         elemsize = dx* dz
      #         print ('Element size, dx, dz (all in m): ', elemsize, dx, dz)
      #         # STF
      #         print ('Computing STF ...')
      #         slip_rates = np.array(self.fault_stf_SR)
      #         slips = np.array(self.faultsta_slip)
      #         STF = np.zeros(t.shape)
      #         moment = np.zeros(t.shape)
      #         for  mu, slip_rate, slip in zip(fault_mu, slip_rates, slips):
      #             STF += slip_rate* elemsize* mu     
      #             moment += slip* elemsize* mu                    
      #         # PADDING
      #         dt = t[1]- t[0]    
      #         if is_padding:
      #             self.t_STF = np.arange(20*len(t))* dt
      #             self.STF = np.zeros(self.t_STF.shape[0])
      #             self.STF[:len(STF)] = STF
      #             self.moment = np.zeros(self.t_STF.shape[0])+ max(moment)
      #             self.moment[:len(moment)] = moment
      #         else:
      #             self.STF = STF
      #             self.t_STF = t
      #             self.moment = moment
      #         #
      #         print ('Computing spectrum ...')
      #         # Spectrum  
      #         df=1.0/ dt
      #         N = len(self.t_STF)
      #         _f = np.linspace(0.0, 1.0/(2.0*dt), N//2)
      #         spec = abs(fft.fft(self.STF))* dt 
      #         specMoment = abs(fft.fft(self.moment ))* dt       
      #         self.specf = _f[1:]
      #         self.specD = spec[:int(N/2)-1]   
      #         self.specMoment = specMoment[:int(N/2)-1]     
      #         Mw = (np.log10(self.specD[0])- 9.1)/ 1.5
      #         print('Mw: ', Mw)
      #         self.M0 = self.specD[0]
      #         self.Mw = Mw        
      #         #
      #         print ('Smoothening by konno-ohmachi ...')
      #         # smooth by konno-ohmachi
      #         specD_sm = self.specD
      #         if is_smooth:
      #             cdt =  (self.specf <= fmax)
      #             f = self.specf[cdt][::jump_ko]
      #             specD_sm = ko(self.specD[cdt][::jump_ko], df, pow_ko)      
      #             self.specD_sm = specD_sm
      #             self.specf_sm = f        
      #         # brune's
      #         fc1, fc2, spec_JA19 = get_JA19_spectrum(Mw=self.Mw, f=self.specf)
      #         self.specD_ja = self.specD[0]*spec_JA19
      #         fc = (fc1* fc2)** 0.5
      #         print ('fc: ', fc)                            
      #         self.brune = self.specD[0]/(1.0+(self.specf/fc)**2)
      #         self.fc = np.array([fc,fc1,fc2])
      #         #
      #         print ('Pickling ...')
      #         try: os.makedirs(rep_store)
      #         except: pass      
      #         np.save(rep_store+'/t_STF.npy', self.t_STF)
      #         np.save(rep_store+'/STF.npy', self.STF)
      #         np.save(rep_store+'/moment.npy', self.moment )              
      #         np.save(rep_store+'/specMoment.npy', self.specMoment)                            
      #         np.save(rep_store+'/specf_sm.npy', self.specf_sm)
      #         np.save(rep_store+'/specD_sm.npy', self.specD_sm)
      #         np.save(rep_store+'/specD.npy', self.specD)              
      #         np.save(rep_store+'/specf_ja.npy', self.specf)
      #         np.save(rep_store+'/specD_ja.npy', self.specD[0]*spec_JA19)
      #         np.save(rep_store+'/specD_brune.npy', self.brune)
      #         np.save(rep_store+'Mw.npy', np.array([self.Mw]))
      #         np.save(rep_store+'fc.npy', self.fc)    
      #       except:
      #         print ('prerequisite: read_all_fault_stations')  
      #         print ('input format: get_magnitude(self,all_x,all_z,t,all_SR,fault_mu,plot_source=True,jump_ko=10,\
      #                     pow_ko=20.0,fmax=10.0, \
      #                     is_smooth=True,is_padding=False)')
      #     else:
      #         print ('Opening pickle box ...')
      #         self.t_STF = np.load(rep_store+'/t_STF.npy')
      #         self.STF = np.load(rep_store+'/STF.npy')
      #         self.moment = np.load(rep_store+'/moment.npy')                            
      #         self.specMoment = np.load(rep_store+'/specMoment.npy')              
      #         if is_smooth: self.specf_sm = np.load(rep_store+'/specf_sm.npy')
      #         if is_smooth: self.specD_sm = np.load(rep_store+'/specD_sm.npy')
      #         self.specD = np.load(rep_store+'/specD.npy')
      #         self.specf = np.load(rep_store+'/specf_ja.npy')
      #         self.specD_ja = np.load(rep_store+'/specD_ja.npy')
      #         self.brune = np.load(rep_store+'/specD_brune.npy')
      #         self.Mw = np.load(rep_store+'/Mw.npy')
      #         self.fc = np.load(rep_store+'/fc.npy')
      #         print('Mw: ', self.Mw)
      #     print('Done!')
      #     if plot_source:
      #         plt.close('all')
      #         print ('Plotting ...')
      #         plt.subplot(221)
      #         plt.xlim(-1,20)
      #         plt.plot(self.t_STF,self.STF,'k',lw=1)
      #         plt.grid() 
      #         plt.xlabel('Time (s)'); plt.ylabel('Moment rate (Nm/s)')
      #         #
      #         plt.subplot(222)
      #         plt.xlim(1e-3,fmax)
      #         try: plt.loglog(self.specf_sm, self.specD_sm,'k')    
      #         except: plt.loglog(self.specf, self.specD,'k')
      #         plt.loglog(self.specf, self.brune, c='royalblue', lw=1.0, linestyle='-.', label='w2 model')
      #         plt.loglog(self.specf, self.specD_ja, c='royalblue', lw=1.0, linestyle=':', label='JA19_2S')
      #         plt.axvline(x=self.fc[1], c='k', linestyle=':', alpha=0.5)
      #         plt.axvline(x=self.fc[2], c='k', linestyle=':', alpha=0.5)
      #         plt.xlabel('Frequency (Hz)')
      #         #
      #         plt.subplot(223)
      #         plt.plot(self.t_STF,self.moment,'k',lw=1)
      #         plt.grid()
      #         #
      #         plt.subplot(224)
      #         plt.loglog(self.specf,self.specMoment,'k')              
      #         plt.grid()
      #         #
      #         plt.show()
      #   ##

      def plot_rupture_time(self, constep=1.0,contour=False,cmap='Blues',Ncmap=6, \
                            tmax=0.0,xy_lim=None,vmin=0,vmax=10,**kwargs):
            import pandas as pd
            # read file
            try:
              # only for a single fault
              fname = self.directory+'/RuptureTime_Fault1'
              data = pd.read_csv(fname, names=('x','z','trupt'), \
                             delim_whitespace=True, header=10)      
              Trupt = data['trupt'].values
              Trupt_not_overwritten = Trupt.copy()
              X = data['x'].values/1.e3
              Z = data['z'].values/-1.e3
            except:
              print ('No RuptureTime_Fault file, reading from snapshots ...')
              Trupt = self.fault['Trup'][-1] # from last snapshot file
              X, Z = self.fault['x'], self.fault['z'] 
              print ('Done!')
            else:
              print ('ERROR in reading file!')
            if tmax>0.0 : vmax = tmax
            if 'tmax' in kwargs:  vmax = kwargs['tmax']
            print ('Max of rupture time (s): ', '%.1f' % (vmax))
            # for plotting purposes
            Trupt[Trupt >= vmax] = 0.0
            ext = [min(X), max(X), min(Z), max(Z)]
            xi, zi = np.meshgrid(np.linspace(ext[0],ext[1],1000),np.linspace(ext[2],ext[3],1000))
            y = gd( (X,Z), Trupt, (xi,zi), method='linear')
            y = np.flipud(y)
            #
            fig = plt.figure(figsize=(6,3)); set_style(whitegrid=False)
            ax = fig.add_subplot(111)
            ax.set_xlabel('Along strike (km)')
            ax.set_ylabel('Along dip (km)')
            if xy_lim==None: xy_lim=ext
            ax.set_xlim(xy_lim[0], xy_lim[1]); ax.set_ylim(xy_lim[2], xy_lim[3])
            try: ax.scatter(self.x_hypo, self.z_hypo, marker='*', color='r', s=100)
            except: pass
            cmap = plt.cm.get_cmap(cmap, Ncmap) 
            im = ax.imshow(y, extent=ext,vmin=vmin, vmax=vmax, cmap=cmap,aspect='auto')
            levels = np.linspace(vmin, vmax, num=int(vmax/constep))
            if contour: ax.contour(y, levels, colors='snow', alpha=0.5, extent=ext, origin='upper')
            c = plt.colorbar(im, fraction=0.046, pad=0.1,shrink=0.75)
            c.set_label('Rupture time (s)')
            c.mappable.set_clim(vmin, vmax)
            plt.tight_layout()
            plt.show()        
            return 
      ##

      def get_rupture_speed(self,info=False,eps=0.5,is_Vrup_pickled=True,rep_store='./STORE/', \
                              plot_rupture_speed=True,cmap='RdBu_r',Ncmap=6,xy_lim=None,\
                              Nsample=1000,normalise_by_Vs=False,vmin=0.0,vmax=2.0,\
                              aspect='auto'):
          if not is_Vrup_pickled:
              Vruptlist = []
              X, Z, Trupt = self.fault['x'], self.fault['z'], self.fault['Trup'][-1] # from last snapshot file
              Vs = self.fault['Vs']/1e3 # km/s
              # ~ to neglect nucleation process and non-broken parts
              cdt = (Trupt > eps) 
              print ('Trupt[cdt].shape', Trupt[cdt].shape)
              for i, (_trup, _Vs) in enumerate( zip(Trupt[cdt], Vs[cdt]) ):
                  _x = X[cdt][i]
                  _z = Z[cdt][i]
                  if info: print('i, _x, _z, _trup', i, _x, _z, _trup)
                # to avoid too close points on nearly the same time contour
                  cdt1 = (Trupt < _trup- 0.25) 
                  points_behind = (X[cdt1]-_x)**2+ (Z[cdt1]-_z)**2
                  cdt2 = (points_behind == min(points_behind))
                  if info: print ('closest point: ', X[cdt1][cdt2], Z[cdt1][cdt2]  )
                  dist = points_behind[cdt2]**0.5    
                  if info: print ('dist (km)', dist)        
                  delta_t = _trup- Trupt[cdt1][cdt2]    
                  if info: print ('delta_t (s)', delta_t)
                  Vruptlist.append ( max( dist/delta_t ) )
                  if info: print ('Vrupt (km/s), Vrupt/Vs found: ',  min(dist/delta_t),  min(dist/delta_t)/_Vs)
                  if info: print ('*')
                  #
                  if  max( dist/delta_t ) > 10.0:
                      print ('*')
                      print ('evaluated point: ', _x, _z, _trup)
                      print ('closest point: ', X[cdt1][cdt2], Z[cdt1][cdt2]  )
                      print ('dist', dist)
                      print ('delta_t', delta_t)
                      print ('Vrupt found: ',  (dist/delta_t))
                      print ('*')
              Vrupt = np.zeros(Trupt.shape)
              Vrupt[cdt] = np.array(Vruptlist)   
              self.Vrup = Vrupt # in km/s
              self.Vrup_normalised = self.Vrup/ Vs
              print ('Pickling ...')
              try: os.makedirs(rep_store)
              except: pass        
              np.save(rep_store+'/Vrup.npy', self.Vrup)
              np.save(rep_store+'/Vrup_normalised.npy', self.Vrup_normalised)
          else:
              print('Opening pickle box ...')
              print ('in the folder ', rep_store)
              try:
                  self.Vrup = np.load(rep_store+'/Vrup.npy')
                  self.Vrup_normalised = np.load(rep_store+'/Vrup_normalised.npy') 
              except:
                  print('No \'Vrup.npy\' or \'Vrup_normalised.npy\' data found?')
          print ('Rupture speed ready!')
          
          if plot_rupture_speed:
              X, Z = self.fault['x'], self.fault['z']
              cmap = plt.cm.get_cmap(cmap, Ncmap)    
              #
              data = self.Vrup
              if normalise_by_Vs: data = self.Vrup_normalised
              ext = [min(X), max(X), min(Z), max(Z)]   
              xi, zi = np.meshgrid(np.linspace(ext[0],ext[1],Nsample),np.linspace(ext[2],ext[3],Nsample))
              y = gd( (X,Z), data, (xi,zi), method='linear')
              y = np.flipud(y)
              #
              print ('Plotting ...')
              plt.close('all')
              fig = plt.figure(figsize=(6,3))
              ax = fig.add_subplot(111)
              ax.set_xlabel('Along strike (km)')
              ax.set_ylabel('Along dip (km)')
              if xy_lim==None: xy_lim=ext
              ax.set_ylim(xy_lim[2],xy_lim[3])
              ax.set_xlim(xy_lim[0],xy_lim[1])
              try: ax.scatter(self.x_hypo, self.z_hypo, marker='*', color='r', s=100)
              except: pass
              im = ax.imshow(y, extent=ext, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect)
              c = plt.colorbar(im, fraction=0.046, pad=0.1,shrink=0.75)
              if normalise_by_Vs: c.set_label('Rupture speed (Vs)')
              else: c.set_label('Rupture speed (km/s)')
              c.mappable.set_clim(vmin, vmax)
              plt.tight_layout()
              plt.show()          
          #            
          return 
      ##


      def plot_stress_drop(self,Nsample=1000,cmap='inferno',Ncmap=4,vmin=0,vmax=20):
          ''' calculates and plots stress drop by using snapshot data.'''

          try:
              from scipy.interpolate import griddata as gd

              # stress drop = initial stress - final stress 
              self.stress_drop = self.fault['Tx'][0]- self.fault['Tx'][-1]
              self.stress_drop /= 1e6  # MPa
              # average stress drop on ruptured parts
              cdt = (self.stress_drop > 0.0)
              data = self.stress_drop[cdt]
              X = self.fault['x'][cdt] 
              Z = self.fault['z'][cdt]  
              print ('Average stress drop (MPa): ', np.average(data))
              print ('Max point location (x,z): ',X [data==max(data)], Z [data==max(data)])
              #  interpolate
              ext = [min(X), max(X), min(Z), max(Z)]
              xi, zi = np.meshgrid(np.linspace(ext[0],ext[1], Nsample), np.linspace(ext[2],ext[3],Nsample))
              y = gd( (X,Z), data, (xi,zi), method='linear')
              y = np.flipud(y) 
              #
              fig = plt.figure(figsize=(6,3))
              ax = fig.add_subplot(111, aspect='auto')
              ax.set_xlabel('Along strike (km)')
              ax.set_ylabel('Along dip (km)')    
              cmap = plt.cm.get_cmap(cmap, Ncmap)    # 11 discrete colors
              #
              im = ax.imshow(y, extent=ext, vmin=vmin, vmax=vmax, cmap=cmap)
              c = plt.colorbar(im, fraction=0.046, pad=0.05, shrink=0.5, label='Stress drop (MPa)')
              c.mappable.set_clim(vmin, vmax)
              plt.grid()
          except:
              print ('prerequisite:  self.fault data')
              print ('input format: plot_stress_drop(self,Nsample=1000,cmap=\'inferno\',Ncmap=4,vmin=0,vmax=20)')
      ##


      def plot_rupture_front(self, **kwargs):

            # Final state
            Trupt = self.fault['Trup'][-1]
            Tproz = self.fault['Tproz'][-1]
            X = self.fault['x'] 
            Y = self.fault['y']  
            Z = self.fault['z']  

            # strike-slip fault
            # from top
            # Only the horizontal fault line
            X = X [ (Z == 0.0) ]
            Trupt = Trupt [ (Z == 0.0)]
            Tproz = Tproz [ (Z == 0.0)]
            Xs, Tps = zip(*sorted(zip(X,Tproz)))                  
            Xs, Ts = zip(*sorted(zip(X,Trupt))) 

            max_arrival = X [(Trupt==0.0) & (X > 0.0)]   
            maxpt = max(X)
            if len(max_arrival) > 0: maxpt = min(max_arrival)
            xmax = max(X); xmin = min(X)
            if 'xmax' in kwargs: xmax= kwargs['xmax']
            if 'xmin' in kwargs: xmin= kwargs['xmin']

            ymax = max(max(Ts), max(Tps)); ymin = 0.0
            if 'ymax' in kwargs: ymax = kwargs['ymax']
            if 'ymin' in kwargs: ymin = kwargs['ymin']

            ###
            plt.close('all')        
            fig = plt.figure(); set_style(whitegrid=True, scale=0.8)
            ax = fig.add_subplot(111)
            ax.set_xlabel('Horizontal distance (km)')
            ax.set_ylabel('Time (s)')
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.plot(Xs, Ts, color='k', label='Front')
            ax.plot(Xs, Tps, color='royalblue', label='Tail')
            tit = 'Rupture front reaches to \n a max distance of '+ str('%.1f' % (maxpt))
            ax.set_title(tit)
            ax.legend(loc='best')
            plt.tight_layout(); plt.show()
    ###

      def plot_fault_station_data_from_snaphots(self, xpt=0.0, ypt=0.0, zpt=0.0):

            X = self.fault['x'] 
            Y = self.fault['y']  
            Z = self.fault['z']  
            Dx = self.fault['Dx']   
            Dz = self.fault['Dz']   
            Vx = self.fault['Vx']   
            Vz = self.fault['Vz']   
            Tx = self.fault['Tx']   
            Ty = self.fault['Ty']   
            Tz = self.fault['Tz']   
            Sg = self.fault['Sg']   

            # Nearest station
            dist = [(_x-xpt)**2+ (_y-ypt)**2+ (_z-zpt)**2  for _x,_y,_z in zip(X,Y,Z)]
            idx = np.where(dist == min(dist))[0][0]
            print('Found coordinates (x,y,z): ', X[idx], Y[idx], Z[idx])
            print('Index:                     ', idx)

            plt.close('all')
            fig = plt.figure(figsize=(10,6)); set_style(whitegrid=True, scale=0.8)
            plt.subplots_adjust(hspace=0.45, wspace=0.42, left=0.1, right=0.960)

            ax = fig.add_subplot(231)
            ax.set(title='Strike-slip (m)')
            ax.plot([D[idx] for D in Dx], c='k')

            ax = fig.add_subplot(234)
            ax.set(title='Dip-slip (m)')
            ax.plot([D[idx] for D in Dz], c='k')

            ax = fig.add_subplot(232)
            ax.set(title='Vx (m/s)')
            ax.plot([V[idx] for V in Vx], c='k')

            ax = fig.add_subplot(235)        
            ax.set(title='Vz (m/s)')
            ax.plot([V[idx] for V in Vz], c='k')

            ax = fig.add_subplot(233)        
            ax.set(title='Stress (MPa)')
            ax.plot([T[idx]/1.e6 for T in Tx], label='Tx', color='red')
            ax.plot([T[idx]/1.e6  for T in Ty], label='Ty', color='b')
            ax.plot([-T[idx]/1.e6  for T in Tz], label='-Tz', color='k')
            ax.legend(loc='lower right', bbox_to_anchor=(1.0,-1.4))
            plt.show()
      ###

      def get_and_plot_seismo(self, stations, filtering=False, fmin=0.0, fmax=1.0, 
                                plot_spectogram=False, plot_timehistories=True,
                                is_binary=False, is_acceleration=False,suffix='H'):

         from obspy.core import Trace


         # change spectogram colorbar plot
         for sta in stations:

            plt.close('all')

            # x direction
            fname = self.directory+'/'+sta+'.'+suffix+'XX.semv'
            if is_binary:
               with open(fname,'rb') as f: Vx = np.fromfile(f, dtype=np.float32)
               t = np.arange(len(Vx))* self.dt
            else:
               t = np.genfromtxt(fname, usecols=0)
               Vx = np.genfromtxt(fname, usecols=1)
            tr_x = Trace()
            tr_x.data = Vx
            tr_x.time = t
            tr_x.stats.delta = t[1]-t[0]
            tr_x.stats.starttime = t[0]
            if filtering: tr_x.filter(type='lowpass', freq=fmax, corners=2, zerophase=True)
            if is_acceleration: tr_x.differentiate() 

            # y direction
            fname = self.directory+'/'+sta+'.'+suffix+'XY.semv'
            if is_binary:
               with open(fname,'rb') as f: Vy = np.fromfile(f, dtype=np.float32)
            else:
               Vy = np.genfromtxt(fname, usecols=1)
            tr_y = Trace()
            tr_y.data = Vy
            tr_y.time = t            
            tr_y.stats.delta = t[1]-t[0]
            tr_y.stats.starttime = t[0]
            if filtering: tr_y.filter(type='lowpass', freq=fmax, corners=2, zerophase=True)
            if is_acceleration: tr_y.differentiate() 
               
            # z direction
            fname = self.directory+'/'+sta+'.'+suffix+'XZ.semv'
            if is_binary:
               with open(fname,'rb') as f: Vz = np.fromfile(f, dtype=np.float32)
            else:
               Vz = np.genfromtxt(fname, usecols=1)
            tr_z = Trace()
            tr_z.data = Vz
            tr_z.time = t
            tr_z.stats.delta = t[1]-t[0]
            tr_z.stats.starttime = t[0]
            if filtering: tr_z.filter(type='lowpass', freq=fmax, corners=2, zerophase=True)
            if is_acceleration: tr_z.differentiate() 
               
            if plot_timehistories:
               fig = plt.figure(figsize=(8,4)); #set_style(whitegrid=True, scale=0.8)
               plt.subplots_adjust(top=0.8)

               ax = fig.add_subplot(311)
               tit = 'Velocity-time histories (m/s)'
               if is_acceleration: tit='Acceleration (m/s/s)'
               ax.set_title(tit)
               ax.set_ylabel('x')
               ax.plot(t, tr_x.data, color='k', lw=0.7)
               ax = fig.add_subplot(312)
               ax.set_ylabel('y')
               ax.plot(t, tr_y.data, color='k', lw=0.7)
               ax = fig.add_subplot(313)
               ax.set_ylabel('z')
               ax.plot(t, tr_z.data, color='k', lw=0.7)
               plt.tight_layout(); plt.show()

         if plot_spectogram:
            plt.close('all')
            tmin, tmax = min(t), max(t)
            print (tmin, tmax)
            fig = plt.figure(); set_style(whitegrid=True, scale=0.7)
            ax1 = fig.add_subplot(321)
            tr_x.spectrogram(axes=ax1)
            ax1.set_ylim(fmin, fmax)
            ax1.set_xlim(tmin, tmax)
            ax1.set_ylabel('Frequency (Hz)')
            ax2 = fig.add_subplot(322)
            mappable = ax1.images[0]
            plt.colorbar(mappable=mappable, cax=ax)

            ax1 = fig.add_subplot(323)
            tr_y.spectrogram(axes=ax1)
            ax1.set_ylim(fmin, fmax)
            ax1.set_xlim(tmin, tmax)         
            ax1.set_ylabel('Frequency (Hz)')
            ax2 = fig.add_subplot(324)
            mappable = ax1.images[0]
            plt.colorbar(mappable=mappable, cax=ax2)

            ax1 = fig.add_subplot(325)
            tr_z.spectrogram(axes=ax1)
            ax1.set_ylim(fmin, fmax)
            ax1.set_xlim(tmin, tmax)
            ax1.set_ylabel('Frequency (Hz)')
            ax1.set_xlabel('Time (s)')
            ax2 = fig.add_subplot(326)
            mappable = ax1.images[0]
            plt.colorbar(mappable=mappable, cax=ax2)

            plt.tight_layout(); plt.show()
            ###

         return tr_x, tr_y, tr_z
         ###


      def plot_fault_station_data(self, filename=None):

         import pandas as pd

         # x: along strike; z: along dip      
         fname = self.directory+ filename
         data = pd.read_csv(fname, names=('t','Dx','Vx','tau_xy', 'Dz', 'Vz', 'tau_yz', 'tau_yy' ), \
                        delim_whitespace=True, header=20)      
         t = data['t'].values
         Dx = data['Dx'].values
         Vx = data['Vx'].values
         tau_xy = data['tau_xy'].values
         Dz = data['Dz'].values
         Vz = data['Vz'].values
         tau_yz = data['tau_yz'].values
         tau_yy = data['tau_yy'].values

         plt.close('all')
         fig = plt.figure(figsize=(10,6)); set_style(whitegrid=True, scale=0.8)
         plt.subplots_adjust(hspace=0.45, wspace=0.42, left=0.1, right=0.960)

         tit = 'Fault station (x,y)   '+ filename+'\n'
         fig.suptitle(tit)
         ax = fig.add_subplot(231)
         ax.set(title='Strike-slip (m)')
         ax.plot(t, Dx, c='k')

         ax = fig.add_subplot(234)
         ax.set(title='Dip-slip (m)')
         ax.plot(t, Dz, c='k')

         ax = fig.add_subplot(232)
         ax.set(title='Vx (m/s)')
         ax.plot(t, Vx, c='k')

         ax = fig.add_subplot(235)        
         ax.set(title='Vz (m/s)')
         ax.plot(t, Vz, c='k')

         ax = fig.add_subplot(233)        
         ax.set(title='Stress (MPa)')
         ax.plot(tau_xy, label=r'$\tau_{xy}$', color='red')
         ax.plot(tau_yz, label=r'$\tau_{yz}$', color='b')
         ax.plot(-tau_yy, label=r'$-\tau_{yy}$', color='k')
         ax.legend(loc='lower right', bbox_to_anchor=(1.0,-1.4))
         plt.show()
      ###



      def plot_slip_along_fault(self, search_tip='/*y_0.0*dat'):

         ''' I use this subroutine to plot surface slip'''

         import glob
         import pandas as pd

         filenames = glob.glob(self.directory+search_tip)
         strike_slip = []; dip_slip = []; position = []

         # get all filenames of fault stations
         for fname in filenames:
            # x: along strike; z: along dip      
            data = pd.read_csv(fname, names=('t','Dx','Vx','tau_xy', 'Dz', 'Vz', 'tau_yz', 'tau_yy' ), \
                        delim_whitespace=True, header=20)  
            
            niter = len(data['Dx'].values)
            Dx = data['Dx'].values[:niter]
            Dz = data['Dz'].values[:niter]
            
            dum = max(abs(Dx))
            strike_slip.append(dum* np.sign(dum))
            dum = max(abs(Dz))
            dip_slip.append(dum* np.sign(dum))    
            
            # where this station is:
            num1 = fname.find('x_');
            num2 = fname.find('_y');
            position.append(float(fname[num1+2:num2]))
         ###

         # Order lists by strike position
         ind = sorted(range(len(position)), key=lambda k: position[k])
         strike_slip = np.array(strike_slip)
         dip_slip = np.array(dip_slip)
         position = np.array(position)
         set_style(whitegrid=True)
         ###

         # Plot
         plt.close('all')
         plt.plot(position[ind], strike_slip[ind], c='red', label='along strike')
         plt.plot(position[ind], dip_slip[ind], c='k', label='along dip')
         plt.xlabel('Fault strike (km)')
         plt.ylabel('Surface slip (m)')
         plt.tight_layout()
         plt.show()
      ###


      def get_ground_motion_data(self,net=None,suffix='C',f_LP=3.0, f_HP=None, infoint=100,\
                             is_pickled=False,rep_store='./STORE/', \
                             is_acceleration=False,ista_beg=0,ista_end=None):

          """reads output seismograms and computes PGV and PGV after LP filtering. 
          stores GMM into dataframe and seismograms in np array."""

          import pandas as pd 

          if not is_pickled:
              # Stations
              fname = self.directory+'/output_list_stations.txt'
              stas = np.genfromtxt(fname,usecols=0,dtype=None,encoding='utf-8')
              nets = np.genfromtxt(fname,usecols=1,dtype=None,encoding='utf-8')
              xx = np.genfromtxt(fname, usecols=2) # along strike
              yy = np.genfromtxt(fname, usecols=3) # off-fault
              zz = np.genfromtxt(fname, usecols=4) 
              # make dataframe for seismograms
              df = pd.DataFrame()
              names = [net+ '.'+ sta for sta, net in zip(stas, nets)]
              df['name'] = names
              df['x'] = xx/ 1e3
              df['y'] = yy/ 1e3
              df['z'] = zz/ 1e3
              # epicentral distance (model 1)
              self.y_hypo = 0.0
              try:
                  _dist = (df['x'].values- self.x_hypo)**2 + (df['y'].values- self.y_hypo)**2
                  df['r_epi'] = _dist** 0.5
              except:
                  print('Hypocenter location is required for r_epi!')
              #
              print ('Computing PGV and PGA ...')
              PGA_pol, PGV_pol, PGAs, PGVs = [], [], [], []
              sismos = []
              pol = np.array( ['x','y','z']) 
              sismos_x, sismos_y, sismos_z = [], [], []
              for i, sta in enumerate(df['name'].values[:]):
                  
                  mod = int(len(df)*infoint/ 100)
      #             print ('i, mod: ', i, mod)
                  if (i % mod == 0): print ('i, % done: ', i, '%.0f' % (i/mod*infoint) )
                      
                  # acceleration x,y,z
                  tr_x, PGVx, PGAx = get_seismo_fast_1component(self, sta, filtering=True, fmax=f_LP, fmin=f_HP,
                                          is_binary=True, is_acceleration=is_acceleration,suffix=suffix, compo='XX.semv')
                  tr_y, PGVy, PGAy  = get_seismo_fast_1component(self, sta, filtering=True, fmax=f_LP, fmin=f_HP,
                                          is_binary=True, is_acceleration=is_acceleration,suffix=suffix, compo='XY.semv')
                  tr_z, PGVz, PGAz  = get_seismo_fast_1component(self, sta, filtering=True, fmax=f_LP, fmin=f_HP,
                                          is_binary=True, is_acceleration=is_acceleration,suffix=suffix, compo='XZ.semv')    
                  _PGAs = np.array( [PGAx, PGAy, PGAz] )
                  _PGA = max(_PGAs)
                  _pol = pol[_PGAs == _PGA]
                  PGAs.append(_PGA)
                  PGA_pol.append(_pol)

                  _PGVs = np.array( [PGVx, PGVy, PGVz] )
                  _PGV = max(_PGVs)
                  _pol = pol[_PGVs == _PGV]
                  PGVs.append(_PGV)
                  PGV_pol.append(_pol)

                  sismos_x.append(tr_x.data) # velocities
                  sismos_y.append(tr_y.data) # velocities
                  sismos_z.append(tr_z.data) # velocities              
              #
              df['PGA (g)'] = [pga/9.81 for pga in PGAs] 
              df['PGV (m/s)'] = PGVs
              df['PGA_polar'] = PGA_pol
              df['PGV_polar'] = PGV_pol
              NT = sismos_x[0].shape[0]
              Nsta = len(sismos_x)
              sismos_store = np.zeros((Nsta, NT, 3))
              sismos_store[:,:, 0] = np.array(sismos_x)
              sismos_store[:,:, 1] = np.array(sismos_y)
              sismos_store[:,:, 2] = np.array(sismos_z)    
              print ('Pickling ...')
              try: os.makedirs(rep_store)
              except: pass
              df.to_pickle(rep_store+'/df_ground_motion')
              np.save(rep_store+'/sismos_store.npy',sismos_store)        
          else:  
              print ('Opening pickle box ...')  
              df = pd.read_pickle(rep_store+'/df_ground_motion')
              sismos_store = np.load(rep_store+'/sismos_store.npy')
          ##
          print ('Done!')
          return df, sismos_store
      ##            
             
    

      def plot_PGV_and_polarisation(self,df,Ncmap=21,cmap='rainbow',vmin=0,vmax=3):

          fig = plt.figure(figsize=(6,6))
          ax = plt.subplot(211)
          cmap = plt.cm.get_cmap(cmap,Ncmap)
          im = plt.scatter(df['x'], df['y'], c=df['PGV (m/s)'], cmap=cmap, vmin=vmin,vmax=vmax)
          plt.scatter(self.x_hypo, 0.0, marker='*', color='snow')
          plt.title('Top view -- PGV (m/s)')
          plt.ylabel('Off-fault distance (km)')
          cbar_ax = fig.add_axes([0.85, 0.6, 0.025, 0.25])
          fig.colorbar(im, cax=cbar_ax)
          ##
          plt.subplot(212, sharex=ax)
          cdt = (df['PGV_polar'] == 'x')
          im = plt.scatter(df[cdt]['x'], df[cdt]['y'], c='peru', label='FP', alpha=0.75)
          #
          cdt = (df['PGV_polar'] == 'y')
          im = plt.scatter(df[cdt]['x'], df[cdt]['y'], c='violet', label='FN', alpha=0.75)
          #
          cdt = (df['PGV_polar'] == 'z')
          im = plt.scatter(df[cdt]['x'], df[cdt]['y'], c='k', label='Z')
          plt.legend(bbox_to_anchor=(1, 0.5))
          plt.hlines(y=0, xmin=-20,xmax=25, color='k')
          plt.scatter(self.x_hypo, 0.0, marker='*', color='snow')
          plt.title('Top view -- PGV polarisation')
          plt.xlabel('Along strike (km)'); plt.ylabel('Off-fault distance (km)')
          fig.subplots_adjust(right=0.8, hspace=0.3)
          #
###
      def plot_PGA_and_polarisation(self,df,Ncmap=21,cmap='rainbow',vmin=0,vmax=0.1):

          fig = plt.figure(figsize=(6,6))
          ax = plt.subplot(211)
          cmap = plt.cm.get_cmap(cmap,Ncmap)
          im = plt.scatter(df['x'], df['y'], c=df['PGA (g)'], cmap=cmap, vmin=vmin,vmax=vmax)
          plt.scatter(self.x_hypo, 0.0, marker='*', color='snow')
          plt.title('Top view -- PGV (m/s)')
          plt.ylabel('Off-fault distance (km)')
          cbar_ax = fig.add_axes([0.85, 0.6, 0.025, 0.25])
          fig.colorbar(im, cax=cbar_ax)
          ##
          plt.subplot(212, sharex=ax)
          cdt = (df['PGA_polar'] == 'x')
          im = plt.scatter(df[cdt]['x'], df[cdt]['y'], c='peru', label='FP', alpha=0.75)
          #
          cdt = (df['PGA_polar'] == 'y')
          im = plt.scatter(df[cdt]['x'], df[cdt]['y'], c='violet', label='FN', alpha=0.75)
          #
          cdt = (df['PGA_polar'] == 'z')
          im = plt.scatter(df[cdt]['x'], df[cdt]['y'], c='k', label='Z')
          plt.legend(bbox_to_anchor=(1, 0.5))
          plt.hlines(y=0, xmin=-20,xmax=25, color='k')
          plt.scatter(self.x_hypo, 0.0, marker='*', color='snow')
          plt.title('Top view -- PGA polarisation')
          plt.xlabel('Along strike (km)'); plt.ylabel('Off-fault distance (km)')
          fig.subplots_adjust(right=0.8, hspace=0.3)
        #
###



      def plot_STF_and_spectrum(self, jump_ko=10,\
                      pow_ko=20.0,fmax=10.0, lims=[-0.1,20.0], \
                      is_smooth=True,is_padding=False, rep_store='./STORE/', is_plot=True):
          ''' computes STF and its spectrum. requires STF and dt. compute spectrum 
          with Brune\'s and Ji and Archuleta's spectra.'''
          from scipy import fft

          print ("*** TO CORRECT: for overstressed nucleation, correct initialpeak in STF! ***")

          # use padding only for spectrum calculation
          dt = self.dt
          self.time = np.arange(len(self.STF))* dt
          if is_padding:
              time = np.arange(20*len(self.STF))* dt
              STF = np.zeros(time.shape)
              STF[:len(self.STF)] = self.STF
          else:
              STF = self.STF
              time = np.arange(len(self.STF))* dt
          #
          print ('Computing spectrum ...')
          # Spectrum  
          df = 1.0/ dt
          N = len(STF)
          _f = np.linspace(0.0, 1.0/(2.0*dt), N//2)
          spec = abs(fft.fft(STF))* dt 
          freq = _f[1:]
          self.spec_moment_rate = spec[:int(N/2)-1]  
          self.freq_source = freq
          self.M0 = self.spec_moment_rate[0]
          self.Mw = (np.log10(self.M0)- 9.1)/ 1.5
          print('Mw: ', self.Mw)    
          #
          if is_smooth:
              print ('Smoothening by konno-ohmachi ...')
              cdt = (freq <= fmax)
              f_sm = freq[cdt][::jump_ko]
              self.spec_moment_rate_sm = ko(self.spec_moment_rate[cdt][::jump_ko], df, pow_ko)      
              self.freq_source_sm = f_sm        
          # brune's and ji & archuleta
          fc1, fc2, spec_JA19 = get_JA19_spectrum(Mw=self.Mw, f=self.freq_source)
          self.specD_ja = self.M0* spec_JA19
          fc = (fc1* fc2)** 0.5
          print ('fc (geometric mean): ', '%.2f' %(fc) )                             
          self.brune = self.M0/ (1.0+(self.freq_source/ fc)**2)
          #
          print ('Pickling ...')
          try: os.makedirs(rep_store)
          except: pass
          np.save(rep_store+'/STF.npy', self.STF)    
          np.save(rep_store+'/time.npy', self.time)    
          np.save(rep_store+'/freq_source.npy', self.freq_source)
          np.save(rep_store+'/spec_brune.npy', self.brune)
          np.save(rep_store+'/spec_JiArchuleta.npy', self.specD_ja)
          np.save(rep_store+'/spec_moment_rate.npy', self.spec_moment_rate)
          try: np.save(rep_store+'/spec_moment_rate_sm.npy', self.spec_moment_rate_sm)
          except: pass
          try: np.save(rep_store+'/freq_source_sm.npy', self.freq_source_sm)
          except: pass    
          print ('*')    
          #
          if is_plot:
              plt.close('all')
              print ('Plotting ...')
              plt.figure(figsize=(8,4))
              plt.subplot(121)
              plt.xlim(lims[0],lims[1])
              plt.plot(self.time, self.STF,'k',lw=1)
              plt.grid() 
              plt.xlabel('Time (s)'); plt.ylabel('Moment rate (Nm/s)')
              #
              plt.subplot(122)
              plt.xlim(1e-3,fmax)
              try: plt.loglog(self.freq_source_sm, self.spec_moment_rate_sm,'k', lw=1)    
              except: plt.loglog(self.freq_source, self.spec_moment_rate, 'k', lw=1)
              plt.loglog(self.freq_source, self.brune, c='gray', lw=1.0, linestyle='-.', label='w2 model')
              plt.loglog(self.freq_source, self.specD_ja, c='gray', lw=1.0, linestyle=':', label='JA19_2S')
              plt.axvline(x=fc1, c='pink', linestyle='-',lw=2, alpha=0.5)
              plt.axvline(x=fc2, c='pink', linestyle='-',lw=2, alpha=0.5)
              plt.xlabel('Frequency (Hz)')
              plt.grid(True, which="both", ls=":", alpha=0.5)
              plt.tight_layout()
              plt.show()
        ###