# by Elif Oral (elifo@caltech.edu)
# Scripts for Ridgecrest analyses


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from Class_SPECFEM3D import get_FFT, ko

##

def write_out_velocity_profile_data(my_Vs, my_Vp, info=True,rep=None):    
    from scipy.io import FortranFile   
    #
    try: os.makedirs(rep)
    except: pass    
    #
    for  iproc, (_vs, _vp) in enumerate(zip(my_Vs, my_Vp)):
        nglob = len(_vs)
        if info: print ('Processor = ', iproc)    
        if info: print ('Number of points = ', nglob)   
        fout = rep+ 'proc'
        fout += '%06d'% iproc+ '_user_velocity_profile2.bin'
        if info: print ('Filename = ', fout)   
        print ('Writing ...')
        f = FortranFile(fout, 'w')
        f.write_record(np.array([nglob], dtype=np.int32))
        f.write_record(np.array(_vs).reshape((nglob,1)).T)
        f.write_record(np.array(_vp).reshape((nglob,1)).T)
        f.close()
        print ('done!')    
        print (' ')
    print('*')
    return
###


def read_specfem3d_database_coords(iproc=None,rep=None):
    from scipy.io import FortranFile
    #
    fname = rep+'proc'+'%06d' % (iproc)+'_x.bin'
    f = FortranFile(fname, 'r' )
    xproc = f.read_reals( dtype='float32' )
    fname = rep+'proc'+'%06d' % (iproc)+'_y.bin'
    f = FortranFile(fname, 'r' )
    yproc = f.read_reals( dtype='float32' )
    fname = rep+'proc'+'%06d' % (iproc)+'_z.bin'
    f = FortranFile(fname, 'r' )
    zproc = f.read_reals( dtype='float32' )
    print ('iproc: ', iproc)
    print ('len(xproc), len(yproc), len(zproc):', len(xproc), len(yproc), len(zproc))
    print ('*')
    return xproc, yproc, zproc
###


def plot_2d_slice(xx,zz,data,Nx=500,Nz=500,cmap='rainbow_r',Ncmap=11,vmin=2.4e3,vmax=3.8e3,\
                  x_hypo=0.0,d_hypo=0.0,xlim=None,ylim=None):
    from scipy.interpolate import griddata as gd
    import colorcet

    cmap = plt.cm.get_cmap(cmap, Ncmap)    # 11 discrete colors
    ext = [min(xx), max(xx), min(zz), max(zz)]
    xi, zi = np.meshgrid(np.linspace(ext[0],ext[1], Nx), np.linspace(ext[2],ext[3],Nz))
    y = gd( (xx,zz), data, (xi,zi), method='linear')

    fig = plt.figure(figsize=(8,3))
    im = plt.imshow(y, extent=ext,vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', aspect='auto')
    plt.scatter(x_hypo, -d_hypo, marker='*', c='k')

    c = plt.colorbar(im, fraction=0.046, pad=0.05, shrink=0.75, label='Vs (m/s)')
    c.mappable.set_clim(vmin, vmax)
    plt.ylim(ylim[0],ylim[1]); plt.xlim(xlim[0],xlim[1])
###


def get_FFT_data(sismos_store=None,sta_index=[],dt=None,fmax_ko=None,jump=1,pow_ko=20.0,is_smooth=True,\
                rep_store='./STORE/',is_stored=False):
    ''' requires sismos_store and computes and stores FFT of given time histories.'''

    import numpy as np

    if not is_stored:
        FFT_xlist, FFT_ylist, FFT_zlist = [], [], []
        FFT_xlist_sm, FFT_ylist_sm, FFT_zlist_sm = [], [], []
        # station loop
        for ista in sta_index:
            # time data
            Vx, Vy, Vz = sismos_store[ista,:,0], sismos_store[ista,:,1], sismos_store[ista,:,2]
            # FFT
            f, FFT_x = get_FFT(dt=dt, data=Vx)
            f, FFT_y = get_FFT(dt=dt, data=Vy)
            f, FFT_z = get_FFT(dt=dt, data=Vz)
            # local store 
            FFT_xlist.append(FFT_x)
            FFT_ylist.append(FFT_y)
            FFT_zlist.append(FFT_z)    
            # smooth by ko
            if is_smooth:
                cdt =(f <= fmax_ko)    
                f_sm = f[cdt][::jump]
                df_sm = f_sm[1]- f_sm[0]
                FFT_x_sm = ko(FFT_x[cdt][::jump], df_sm, pow_ko)  
                FFT_y_sm = ko(FFT_y[cdt][::jump], df_sm, pow_ko)  
                FFT_z_sm = ko(FFT_z[cdt][::jump], df_sm, pow_ko)  
                #local store
                FFT_xlist_sm.append(FFT_x_sm)
                FFT_ylist_sm.append(FFT_y_sm)
                FFT_zlist_sm.append(FFT_z_sm)    
        #
        NF = FFT_xlist[0].shape[0]
        nsta = len(FFT_xlist)
        # store raw data
        FFT_store = np.zeros((nsta, NF, 3))
        FFT_store[:,:,0] = np.array(FFT_xlist)
        FFT_store[:,:,1] = np.array(FFT_ylist)
        FFT_store[:,:,2] = np.array(FFT_zlist)
        # store smooth data
        if is_smooth:
            NF = FFT_xlist_sm[0].shape[0]        
            FFT_sm_store = np.zeros((nsta, NF, 3))
            FFT_sm_store[:,:,0] = np.array(FFT_xlist_sm)
            FFT_sm_store[:,:,1] = np.array(FFT_ylist_sm)
            FFT_sm_store[:,:,2] = np.array(FFT_zlist_sm)
        #            
        # write store
        np.save(rep_store+'/FFT_store.npy', FFT_store)
        np.save(rep_store+'/FFT_store_freq.npy', f)        
        if is_smooth: 
            np.save(rep_store+'/FFT_sm_store.npy', FFT_sm_store)
            np.save(rep_store+'/FFT_sm_store_freq.npy', f_sm) 
            return f, FFT_store, f_sm, FFT_sm_store
        else:
            return f, FFT_store
    else:
        print ('Opening pickle box ...')
        FFT_store = np.load(rep_store+'/FFT_store.npy')
        f = np.load(rep_store+'FFT_store_freq.npy')
        try: 
            FFT_sm_store = np.load(rep_store+'/FFT_sm_store.npy')
            f_sm = np.load(rep_store+'FFT_sm_store_freq.npy')            
            print ('Done!')
            return f, FFT_store, f_sm, FFT_sm_store
        except: 
            print('No smooth FFT found!')
            return f, FFT_store       
###


def plot_fabians_stransform(data=None,dt=None,tit='',vmin=-3,vmax=4,tmin=0.0,tmax=20.0, \
                            fmin=0.0,fmax=5.0,
                            cmap=None,figname='',y_hline=None,\
                            fig_store='./FIGURES/',fig_name='Stransform.pdf'):
    
    import numpy             as np
    import matplotlib.pyplot as p
    import seaborn           as sns
    import librosa
    import os 
   
    t = np.linspace(0, len(data)*dt, num=len(data))
    print ('plot_fabians_stransform: min, max velocity: ', min(data), max(data))
        
    # Time-window parameters
    tw = 512* dt* 2
    # Computing STFT spectrogram
    sr = 1.0/ dt
    win = int(tw/ dt)
    nfft = 2**(win.bit_length()+ 0)
    hop = int((tw/4) / dt)
    D = np.abs( librosa.stft(data, n_fft=nfft, win_length=win, hop_length=hop) )
    print('min max of D', D.min(), D.max())    
    print('min max of log10(D)', np.log10(D).min(), np.log10(D).max())        
    p.rcParams['font.size']        = 12
    p.rcParams['mathtext.fontset'] = 'stix'
    p.rcParams['font.family']      = ['STIXGeneral']
    fig = p.figure(figsize=(8,8/1.618))      
    p.subplots_adjust(left=0.2, bottom=0.1, right=0.85, top=0.9, hspace=0.3, wspace=0.3)
    sns.set_style('ticks')
    #
    ax = p.subplot2grid((3,1), (0,0), rowspan=1, colspan=1)
    p.suptitle(fig_name)
    p.plot(t, data, lw=0.5, color='k')
    p.xlim(tmin, tmax)
    p.ylabel('Velocity \n (m/s)')
    p.title(tit, fontsize=12)
    p.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    p.setp(ax.get_xticklabels(), visible=False)
    #
    ax = p.subplot2grid((3,1), (1,0), rowspan=2, colspan=1)
    im = p.imshow(np.log10(D), interpolation='bilinear', origin='lower', aspect='auto', \
             vmin=vmin, vmax=vmax, \
             extent=(t.min(),t.max(),0,50), cmap=cmap)
    p.xlim(tmin, tmax); p.ylim(fmin, fmax)
    p.xlabel('Time (s)'); p.ylabel('Frequency (Hz)')    
    if y_hline != None: p.axhline(y=y_hline, linestyle=':',color='k')
    # cbar
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.25])
    fig.colorbar(im, cax=cbar_ax, format='%.1f',shrink=0.75,label='log(|STFT|) (m)')
    sns.despine()    
    try: os.makedirs(fig_store)
    except: pass
    print ('Saving figure as ', fig_store+fig_name+'.pdf')
    fig.savefig(fig_store+fig_name+'.pdf')  
    p.close()
    return 
###


def get_observation(st, df_obs, sta=None, cha=None):
    cdt = (df_obs['Station'] == sta) & (df_obs['Channel'] == cha)
    ista_obs = df_obs[cdt]['Index'].values[0]
    tr = st[ista_obs]
    print ('ista_obs', ista_obs, tr.stats.station, tr.stats.channel)
    return  tr.times(), tr.data
###


def plot_comparison_2_recordings(sismos_store,df_selected,t=None,\
                                 st_recordings=None,f_LP=3,f_HP=1,\
                                 angle=50.0,rotate=True,\
                                 fig_store='./FIGURES/', fig_name='figure',\
                                 xlim=None,t_shift=0):
    from obspy import read
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
   
    # PREPARE RECORDINGS
    # filter
    st = read(st_recordings)
    st.filter('highpass', freq=f_HP) 
    st.filter('lowpass', freq=f_LP)
    # 
    df_obs = pd.DataFrame()
    df_obs['Station'] = [tr.stats.station for tr in st]
    df_obs['Channel'] = [tr.stats.channel[2] for tr in st]
    df_obs['Index'] = [ii for ii, tr in enumerate(st)]    
    # COMPARE
    for ista in df_selected.index:
        print (ista, df_selected.name[ista])    
        name = df_selected.name[ista][3:]    
        #get synthetics
        Vx, Vy, Vz = sismos_store[ista,:,0],sismos_store[ista,:,1],sismos_store[ista,:,2]
        # get recordings
        to, Vobs_E =  get_observation(st, df_obs, sta=name, cha='E')
        to, Vobs_N =  get_observation(st, df_obs, sta=name, cha='N')
        to, Vobs_Z =  get_observation(st, df_obs, sta=name, cha='Z')  
        to -= t_shift
        # rotate
        FP, FN = Vobs_E, Vobs_N
        if rotate:
            FN =  np.cos(np.deg2rad(angle))* Vobs_E+  np.sin(np.deg2rad(angle))* Vobs_N
            FP =  np.cos(np.deg2rad(angle))* Vobs_E-  np.sin(np.deg2rad(angle))* Vobs_N    
        # plot
        fig_name = 'Velocities_compare_real_'+name
        fig = plt.figure(figsize=(8,4))
        plt.subplot(311)
        plt.xlim(xlim[0],xlim[1])
        plt.plot(t, Vx, 'r', lw=1.5)
        plt.plot(to, FP, 'k', lw=0.5)
        plt.grid()
        #
        plt.subplot(312)
        plt.xlim(xlim[0],xlim[1])
        plt.plot(t, Vy, 'r', lw=1.5)
        plt.plot(to, FN, 'k', lw=0.5)
        plt.grid()
        #
        plt.subplot(313)
        plt.xlim(xlim[0],xlim[1])
        plt.plot(t, Vz, 'r', lw=1.5)
        plt.plot(to, Vobs_Z, 'k', lw=0.5)
        plt.grid()
        #
        plt.suptitle('Station '+ name)
        plt.tight_layout()
        try: os.makedirs(fig_store)
        except: pass        
        fig.savefig(fig_store+fig_name+'.pdf')
        plt.close('all')
###


def plot_time_comparison_2_synthetics(df_selected,data1=None, t1=None, data2=None,t2=None,\
                                 fig_store='./FIGURES/', fig_name='figure',\
                                 xlim=None,label1='Model 1',label2='Model 2'):
    from obspy import read
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # COMPARE: index is different than time-dataframe!
    for ista in df_selected.index:
        name = df_selected.name[ista][3:]         
        print (ista, name)    
        #get synthetics
        data_x1, data_y1, data_z1 = data1[ista,:,0], data1[ista,:,1], data1[ista,:,2]
        data_x2, data_y2, data_z2 = data2[ista,:,0], data2[ista,:,1], data2[ista,:,2]        
        # plot
        fig_name = 'Velocity_compare_other_model_'+name
        fig = plt.figure(figsize=(8,4))
        plt.subplot(311)
        plt.xlim(xlim[0],xlim[1])
        plt.plot(t1, data_x1, 'royalblue', lw=1)
        plt.plot(t2, data_x2, 'r', lw=1)
        plt.grid()
        plt.xlabel('Time (s)')  
        #
        plt.subplot(312)
        plt.xlim(xlim[0],xlim[1])
        plt.plot(t1, data_y1, 'royalblue', lw=1)
        plt.plot(t2, data_y2, 'r', lw=1)
        plt.grid()
        plt.xlabel('Time (s)')  
        plt.ylabel('Velocity (m/s)')            
        #
        plt.subplot(313)
        plt.xlim(xlim[0],xlim[1])
        plt.plot(t1, data_z1, 'royalblue', lw=1, label=label1)
        plt.plot(t2, data_z2, 'r', lw=1, label=label2)
        plt.xlabel('Time (s)')  
        plt.legend()
        plt.grid()
        #
        plt.suptitle('Station '+ name)
        try: os.makedirs(fig_store)
        except: pass          
        fig.savefig(fig_store+fig_name+'.pdf')
        plt.close('all')
###    


def plot_FFT_comparison_2_recordings(FFT_sm,df_selected,f=None,\
                                 st_recordings=None,f_LP=3,f_HP=1,\
                                 angle=50.0,rotate=True,\
                                 fig_store='./FIGURES/', fig_name='figure',\
                                 xlim=None,ylim=(1e-5,1e1),twin=None,t_shift=0, \
                                 is_smooth=True,jump=1,\
                                 fmax_ko=10.0,pow_ko=20):
    from obspy import read
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
   
    # PREPARE RECORDINGS
    # filter
    st = read(st_recordings)
    # st.filter('highpass', freq=f_HP) 
    # st.filter('lowpass', freq=f_LP)
    # 
    df_obs = pd.DataFrame()
    df_obs['Station'] = [tr.stats.station for tr in st]
    df_obs['Channel'] = [tr.stats.channel[2] for tr in st]
    df_obs['Index'] = [ii for ii, tr in enumerate(st)]    
    # COMPARE: index is different than time-dataframe!
    for ista in range(len(df_selected)):
        print (ista, df_selected.name[ista])    
        name = df_selected.name[ista][3:]    
        #get synthetics
        FFT_x, FFT_y, FFT_z = FFT_sm[ista,:,0],FFT_sm[ista,:,1],FFT_sm[ista,:,2]
        # get recordings
        to, Vobs_E =  get_observation(st, df_obs, sta=name, cha='E')
        to, Vobs_N =  get_observation(st, df_obs, sta=name, cha='N')
        to, Vobs_Z =  get_observation(st, df_obs, sta=name, cha='Z')  
        to -= t_shift
        cdt = (to >= twin[0]) & (to <= twin[1]) 
        # rotate
        FP, FN = Vobs_E[cdt], Vobs_N[cdt]
        if rotate:
            FN =  np.cos(np.deg2rad(angle))* Vobs_E[cdt]+  np.sin(np.deg2rad(angle))* Vobs_N[cdt]
            FP =  np.cos(np.deg2rad(angle))* Vobs_E[cdt]-  np.sin(np.deg2rad(angle))* Vobs_N[cdt]    
        # FFT    
        fo_x, FFTo_x = get_FFT(dt=to[1]-to[0], data=FP)
        fo_y, FFTo_y = get_FFT(dt=to[1]-to[0], data=FN)
        fo_z, FFTo_z = get_FFT(dt=to[1]-to[0], data=Vobs_Z[cdt])
        # smooth by ko
        cdt2 =(fo_x <= fmax_ko)    
        fo_x_sm = fo_x[cdt2][::jump]
        df_sm = fo_x_sm[1]- fo_x_sm[0]
        # assuming x,y,z arrays have the same len.
        FFT_x_sm = ko(FFTo_x[cdt2][::jump], df_sm, pow_ko)  
        FFT_y_sm = ko(FFTo_y[cdt2][::jump], df_sm, pow_ko)  
        FFT_z_sm = ko(FFTo_z[cdt2][::jump], df_sm, pow_ko)      
        
        # plot
        fig_name = 'FFT_compare_real_'+name
        fig = plt.figure(figsize=(8,4))
        plt.subplot(131)
        plt.xlim(xlim[0],xlim[1]); plt.ylim(ylim[0],ylim[1])
        plt.loglog(f, FFT_x, 'r', lw=1.5)
        plt.loglog(fo_x_sm, FFT_x_sm, 'k', lw=0.5)
        plt.grid()
        plt.xlabel('Frequency (Hz)')  
        plt.ylabel('Fourier amplitude (m/s*s)')    
        #
        plt.subplot(132)
        plt.xlim(xlim[0],xlim[1]); plt.ylim(ylim[0],ylim[1])
        plt.loglog(f, FFT_y, 'r', lw=1.5)
        plt.loglog(fo_x_sm, FFT_y_sm, 'k', lw=0.5)
        plt.grid()
        plt.xlabel('Frequency (Hz)')        
        #
        plt.subplot(133)
        plt.xlim(xlim[0],xlim[1]); plt.ylim(ylim[0],ylim[1])
        plt.loglog(f, FFT_z, 'r', lw=1.5,label='Syn.')
        plt.loglog(fo_x_sm, FFT_z_sm, 'k', lw=0.5, label='Rec.')
        plt.xlabel('Frequency (Hz)')
        plt.legend()
        plt.grid()
        #
        plt.suptitle('Station '+ name)
        try: os.makedirs(fig_store)
        except: pass              
        fig.savefig(fig_store+fig_name+'.pdf')
        plt.close('all')
###    


def plot_FFT_comparison_2_synthetics(df_selected,FFT1=None, f1=None, FFT2=None,f2=None,\
                                 fig_store='./FIGURES/', fig_name='figure',\
                                 xlim=None,label1='Model1',label2='Model2'):
    from obspy import read
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    Nsta = FFT1.shape[0]

    # COMPARE: index is different than time-dataframe!
    for ista in range(Nsta):
        name = df_selected.name[ista][3:] 
        print (ista, name)    
        #get synthetics
        FFT_x1, FFT_y1, FFT_z1 = FFT1[ista,:,0],FFT1[ista,:,1],FFT1[ista,:,2]
        FFT_x2, FFT_y2, FFT_z2 = FFT2[ista,:,0],FFT2[ista,:,1],FFT2[ista,:,2]
        
        # plot
        fig_name = 'FFT_compare_other_models_'+name
        fig = plt.figure(figsize=(8,4))
        plt.subplot(131)
        plt.xlim(xlim[0],xlim[1])
        plt.loglog(f1, FFT_x1, c='royalblue', lw=1.)
        plt.loglog(f2, FFT_x2, c='r', lw=1.0)
        plt.grid()
        plt.xlabel('Frequency (Hz)')  
        plt.ylabel('Fourier amplitude (m/s*s)')    
        #
        plt.subplot(132)
        plt.xlim(xlim[0],xlim[1])
        plt.loglog(f1, FFT_y1, c='royalblue', lw=1)
        plt.loglog(f2, FFT_y2, c='r', lw=1)
        plt.grid()
        plt.xlabel('Frequency (Hz)')        
        #
        plt.subplot(133)
        plt.xlim(xlim[0],xlim[1])
        plt.loglog(f1, FFT_z1, c='royalblue', lw=1, label=label1)
        plt.loglog(f2, FFT_z2, c='r', lw=1, label=label2)
        plt.xlabel('Frequency (Hz)')
        plt.legend()
        plt.grid()
        #
        try: os.makedirs(fig_store)
        except: pass      
        fig.savefig(fig_store+fig_name+'.pdf')
        plt.close('all')
###    


def get_javiers_grid(directory='',mu_value=None,flength=None,fwidth=None,show_fig=False):
    ''' reads outputs of Javier\'s code '''
    filename = directory+ 'tau3d.d'
    xgrid = np.genfromtxt(filename, usecols=0)*1e3- flength*1e3/2.0
    zgrid = np.genfromtxt(filename, usecols=1)*1e3- fwidth*1e3
    tau_xy = np.genfromtxt(filename, usecols=2)/mu_value
    tau_yz = np.genfromtxt(filename, usecols=3)/mu_value
    print ('len(xgrid)', len(xgrid))
    print ('File is read : ', filename); print('***')
    if show_fig:
        fig = plt.figure(figsize=(8,4)); 
        ax = fig.add_subplot(111)
        ax.scatter(xgrid/1e3, zgrid/1e3, s=0.1, c='k')
        ax.set_title('Prefered grid after translation')
        ax.set_xlabel('Along strike (km)'); ax.set_ylabel('Along dip (km)')
        plt.show()
    return xgrid,zgrid,tau_xy,tau_yz
###    


def read_specfem3d_db_files(xjavier=None,zjavier=None,info=True,double_precision=False,\
                    reper='',show_fig=False,nproc=1):
    
    ''' Read fault database (.txt files) of specfem3d'''

    if double_precision: ff = np.float64
    else: ff = np.float32 # this condition is not tested.
    filenames = [reper+'proc'+'%06d'% proc+ '_fault_db.txt' for proc in range(nproc)]
    print ('Reading database files ...'); 
    #
    x_per_proc, y_per_proc, z_per_proc, nglob_per_proc, procnums = [],[],[],[],[]
    for proc, filename in enumerate(filenames):
        with open(filename) as f: # open the file for reading
            iflt = 0
            lines  = f.readlines()
            nfault = int(lines[iflt].split()[2])                           
            _, _, _, _, nspec, nglob, ngll = lines[iflt+2].split()[:]
            nspec = int(nspec); nglob = int(nglob); ngll = int(ngll)
            if nglob > 0:                
                if info: print ('Processor = ', proc)
                if info: print ('Filename = ', filename)                        
                if info: print ('nglob = ', nglob)                        
                i_start = iflt+4
                dummy = lines[i_start:i_start+ 2*nglob] 
                iglobs = [int(dum.split()[0]) for dum in dummy]
                xcoord = [float(dum.split()[1]) for dum in dummy]
                ycoord = [float(dum.split()[2]) for dum in dummy]
                zcoord = [float(dum.split()[3]) for dum in dummy]   
                nglob_per_proc.append(nglob)
                x_per_proc.append(xcoord)
                y_per_proc.append(ycoord)
                z_per_proc.append(zcoord)  
                procnums.append(proc)
                print ('*')
        f.close()      
    print ('Done!'); 
    # flast lists of coordinate
    xcoords = [item for sublist in x_per_proc for item in sublist]
    ycoords = [item for sublist in y_per_proc for item in sublist]
    zcoords = [item for sublist in z_per_proc for item in sublist]
    # Check coordinates shape
    print (len(xcoords), len(ycoords), len(zcoords) )
    print('***')   
    if show_fig:
        # plot (in red: specfem3d; in black:Javier's)
        fig = plt.figure(figsize=(8,4)); 
        ax = fig.add_subplot(111)
        # specfem3d grid
        ax.scatter( np.array(xcoords)/1e3, np.array(zcoords)/1e3, s=0.1, c='r')
        # prefered grid to see if any SEM node is outside the prefered grid:
        try: ax.scatter(xjavier/1e3,zjavier/1e3, s=0.1, c='k')
        except: pass
        ax.set_xlabel('Along strike (km)'); ax.set_ylabel('Along dip (km)')
        ax.set_title('Specfem3d fault grid (in red) vs Javier (in black)')
        plt.show()   
    return xcoords,ycoords,zcoords,x_per_proc, y_per_proc, z_per_proc, nglob_per_proc,procnums
###    


def get_effective_stress_profile(zgrid, show_fig=False, taper_below_cutoff=False,zcreep=15.0,minval=0.5,\
                      is_constant_stress=False,val=100.0):   
    ''' after Shebalin and Narteau (2017), effective stress by depth, suitable for CA.'''
    if is_constant_stress:
        P_eff = [-val for z in zgrid]
    else:
        # pore pressure
        # eff_stress = (1- lambda) litho_stress
        rho_r, rho_w = 2.3e3, 1e3
        L, z_c, eps = 1e3, 2e3, 0.1

        lambda_z = []
        for z in zgrid:
            # assuming max depth value = 0.0 (surface = 0.0)
            _ratio = rho_w/rho_r
            if abs(z) < z_c:
                lambda_z.append(_ratio)
            else:
                _dum = (1.0-eps)+ (_ratio- (1.0-eps))* np.exp((z_c-abs(z))/ L)
                lambda_z.append(_dum)
        #
        P_litho = [rho_r*9.81*z/1e6 for z in zgrid]
        P_pore = [_lam*p for _lam, p in zip(lambda_z, P_litho)]
        P_eff = [(1.0-_lam)*p for _lam, p in zip(lambda_z, P_litho)]    
        #
        if taper_below_cutoff:
            P_eff = np.array(P_eff)
            dist = abs((zcreep*1e3- abs(zgrid)) ) # in m
            cdt = (dist == min(dist))
            val = -P_eff[cdt][0] # value at cut-off depth zcreep
            # exponential decay by depth below cut-off depth
            cdt2 = (abs(zgrid)/1e3 > zcreep)
            P_eff_tapered = np.array(P_eff)
            dist = ( abs(zgrid)/1e3- zcreep ) # positive an in km
            P_eff_tapered[cdt2] = -val*np.exp(-0.5* dist[cdt2])
            P_eff_tapered[abs(P_eff_tapered) < minval ] = -minval
            P_eff = P_eff_tapered
        #            
    if show_fig:
        plt.close('all')
        plt.figure(figsize=(4,4))                
        plt.subplot(111)
        plt.grid()
        try: 
            plt.plot(-np.array(P_litho), -np.array(zgrid)/1e3, c='k', label='Litho')
            plt.scatter(-np.array(P_pore), -np.array(zgrid)/1e3, c='blue', label='Pore', s=1)
        except:
            pass
        plt.scatter(-np.array(P_eff), -np.array(zgrid)/1e3, c='r', label='Eff', s=1)
        plt.legend()
        plt.ylabel('Depth (km)')
        plt.xlabel('Stress (MPa)')
        plt.gca().invert_yaxis()
        fig = plt.gcf()
        plt.show()
    #    
    return P_eff
###


def scale_javiers_stress_values(mu_0=None,mu_s=None,mu_d=None,P_eff=None,tau_xy=None,tau_yz=None):
    # tau_xy_scaled and tau_yz are the parameters 
    # that I use to interpolate and write out.
    tau_xy_scaled, stress_drop_inferred, mu_0_inferred,mu_0_orig = [],[],[],[]
    mu_max_allowed = mu_s- 0.10* (mu_s- mu_d)
    print ('mu_max_allowed: ', mu_max_allowed)        
    # tau_xy: shear stress along strike
    for sigma_n, S_point in zip(P_eff, tau_xy) :
        # (mu_0+ S_point* (mu_0- mu_d) )
        # this operation gives initial shear stress 
        # based on my interpretation of Javier's code's outputs
        # _tau = abs(sigma_n)* (mu_0+ S_point* (mu_0- mu_d) ) 
        #
        # Avoid mu=mu_sta by forcing max to mu_max_allowed
        _mu = min((mu_0+ S_point* (mu_0- mu_d)), mu_max_allowed)
        _tau = abs(sigma_n)* _mu
        mu_0_inferred.append(_mu)    
        tau_xy_scaled.append(_tau)
        mu_0_orig.append(mu_0+ S_point* (mu_0- mu_d))
        _stress_drop = _tau- mu_d* abs(sigma_n)
        stress_drop_inferred.append(_stress_drop)
    #    
    # tau_yz: shear stress along dip
    # tau_yz_scaled = [abs(sigma_n)* (S_point* (_mu_0- mu_d)) for sigma_n, S_point, _mu_0 in zip(sigma_litho,tau_yz,mu_0_inferred )  ]
    # ALSO TO TEST THIS CHANGE!
    tau_yz_scaled = [abs(sigma_n)* (S_point* (mu_0- mu_d)) for sigma_n, S_point in zip(P_eff, tau_yz )  ]
    # Normal stress
    tau_yy = [-abs(sigma_n) for sigma_n in P_eff]
    zzmin, zzmax = min(tau_yz_scaled), max(tau_yz_scaled)
    #
    return np.array(tau_yy),np.array(tau_xy_scaled),np.array(tau_yz_scaled),\
                np.array(stress_drop_inferred),np.array(mu_0_inferred),np.array(mu_0_orig)  
###    


def plot_javiers_scaled_values(xgrid=None,zgrid=None,stress_drop=None,mu0=None,xhypo=None,zhypo=None,Nx=None,Nz=None,\
                          mu_s=None,mu_d=None,tau_xy=None,tau_yz=None):

    ext = [min(xgrid), max(xgrid), min(zgrid), max(zgrid)]
    print ( 'Min and max stress drop ', min(stress_drop), max(stress_drop) )
    print ( 'Average stress drop: ', np.average(stress_drop))
    #
    plt.close('all')
    fig = plt.figure(figsize=(8,6))
    _asp = 'auto';_cmap = 'magma';aspect = 1.0
    #
    ax1 = fig.add_subplot(221, aspect=aspect)
    ax1.scatter(xhypo, zhypo, c='gray', marker='*')
    extkm = [ee/1e3 for ee in ext]
    im1 = ax1.imshow(stress_drop.reshape(Nx,Nz).T, origin='lower', aspect=aspect, extent=extkm,  \
                                  cmap=_cmap, interpolation='bilinear' );
    c1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', fraction=.1, shrink=0.5, label='Stress drop (MPa)')
    c1.mappable.set_clim(0.0, 10.0)
    #
    ax1 = fig.add_subplot(222, aspect=aspect)
    ax1.set_title('Initial stress ratio (mu)')
    ax1.scatter(xhypo, zhypo, c='snow', marker='*')
    im1 = ax1.imshow(mu0.reshape(Nx,Nz).T, origin='lower', aspect=aspect, extent=ext,  \
                              cmap='viridis', interpolation='bilinear' );
    c1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', fraction=.1, shrink=0.75, pad=0.1)
    c1.mappable.set_clim(mu_d, mu_s)
    #
    ax1 = fig.add_subplot(223, aspect=aspect)
    ax1.set_title('Shear stress along strike')
    im1 = ax1.imshow(tau_xy.reshape(Nx,Nz).T, origin='lower', aspect=aspect, extent=ext,  \
                                  cmap=_cmap, interpolation='bilinear' );
    c1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', fraction=.1, shrink=0.25)
    #
    ax1 = fig.add_subplot(224, aspect=aspect)
    ax1.set_title('Shear stress along dip')
    im1 = ax1.imshow(tau_yz.reshape(Nx,Nz).T, origin='lower', aspect=aspect, extent=ext,  \
                                  cmap=_cmap, interpolation='bilinear' );
    c1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', fraction=.1, shrink=0.25)
    plt.tight_layout()    
#     fig.savefig('/Users/elifo/Desktop/stress.png', dpi=300)
###  


# Make a function of 2D interpolation
def make_grid(x=None,z=None,x_db=None,z_db=None):
    points = np.vstack([x, z]).T
    print ('Checking input array sizes: ', x.shape, z.shape, points.shape)
    # make a mesh grid with SEM nodes
    xx = sorted(list(set((x_db))))
    xx = np.array(xx)
    zz = sorted(list(set((z_db))) )
    zz = np.array(zz)
    x_database, z_database = np.meshgrid(xx, zz)
    print ('mesh grid size: ', x_database.shape, z_database.shape)
    return x_database, z_database, points
###


def interpolate_data(x_database, z_database, points, raw_data=None, interpolation='nearest'):
    from scipy.interpolate import griddata
    # input: values; output:values_int_xy
    values = raw_data; values_int = raw_data;
    print ('Interpolation ...')
    if interpolation == 'nearest':
        values_int = griddata(points, values, (x_database, z_database), method='nearest')
    # points: the prefered fault grid (Javier's code's output) that we prepared before SEM simulation
    # values: stress values corresponding to points of Javier's grid
    # (x_database, z_database): 3D code's grid
    # dont set fill_value to get Nan        
    elif interpolation == 'linear':
        value_int_linear = griddata(points, values, (x_database, z_database), method='linear')
        value_int_nearest = griddata(points, values, (x_database, z_database), method='nearest')
        value_int_linear[np.isnan(value_int_linear)] = value_int_nearest[np.isnan(value_int_linear)] 
        values_int = value_int_linear
    print ('Done!')
    return values_int
###


def plot2compare_before_after_interpolation(raw_data=None, int_data=None,interpolation='',N=None,
                                           xgrid=None,zgrid=None,x_db=None,z_db=None):

    fig = plt.figure(figsize=(8,6))
    _asp = 0.75; _cmap = 'magma';aspect = 3
    Nx, Nz = N[0], N[1]
    #
    ext1 = [min(xgrid), max(xgrid), min(zgrid), max(zgrid)]
    ext2 = [np.amin(x_db), np.amax(x_db), np.amin(z_db), np.amax(z_db)]       
    print ('Extent 1:', ext1 )
    print ('Extent 2:', ext2 )    
    ### BEFORE
    ax1 = fig.add_subplot(211, aspect=aspect)
    ax1.set_title('Before interpolation')
    ax1.set_ylim(-25e3,0.0); ax1.set_xlim(-37.5e3,37.5e3)    
    im1 = ax1.imshow(raw_data.reshape(Nx,Nz).T, origin='lower', aspect=_asp, extent=ext1,  \
                                  cmap=_cmap, interpolation=interpolation );
    ### AFTER
    ax2 = fig.add_subplot(212, aspect=_asp)
    ax2.set_title('After interpolation ')
    ax2.set_ylim(-25e3,0.0);ax2.set_xlim(-37.5e3,37.5e3)    
    jump=1
    im2=ax2.scatter(x_db[::jump], z_db[::jump], c=int_data[::jump], cmap='magma',s=0.25)    
    c1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', fraction=.1, shrink=0.5)
    c2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', fraction=.1, shrink=0.5)
    plt.tight_layout(); plt.show()
###


def get_dict_value(dct, number):
    if number in dct.keys(): return dct[number]
    else: print ('** NOT FOUND **', number)  
###      


def write_out_stress_files(write=True, double_precision=False,info=True,nproc=None,reper='',\
                        x_per_proc=None,z_per_proc=None,nglob_per_proc=None,\
                        tau_xy=None,tau_yz=None,tau_yy=None,x_db=None,z_db=None,procnums=None):

    import os
    try: os.makedirs(reper)
    except: print ('Out directory exists!')
    
    x_debug, z_debug, xy_debug, yz_debug, yy_debug = [],[],[],[],[]
    if double_precision: ff=np.float64
    else: ff=np.float32       
    
    # making a dictionary for point-coordinate:stress value to save time
    keys = [(_x,_y) for _x, _y  in  zip(x_db.flatten(), z_db.flatten())] 
    data = tau_xy.flatten()
    dict_tauxy= dict(zip(keys, data))
    #
    data = tau_yz.flatten()
    dict_tauyz= dict(zip(keys, data))
    #
    data = tau_yy.flatten()
    dict_tauyy= dict(zip(keys, data))        
    #
    for nglob, _x_all, _z_all, iproc in zip(nglob_per_proc, x_per_proc, z_per_proc, procnums):
        fout = reper+ 'proc'
        fout += '%06d'% iproc+ '_fault_init_stress.bin'
        if info: print ('Processor = ', iproc)
        if info: print ('Filename = ', fout)   
        # remove repeated coordinates here
        coords_non_repeated = [(x,z) for x, z in zip(_x_all, _z_all)]
        coords_non_repeated = list(dict.fromkeys(coords_non_repeated))
        _x = [x for (x,z) in coords_non_repeated ]
        _z = [z for (x,z) in coords_non_repeated ]
        #
        tau_xy_proc,tau_yz_proc,tau_yy_proc = [],[],[]            
        tau_xy_proc = [get_dict_value(dict_tauxy, (x,z)) for x,z in zip(_x, _z)]
        tau_yz_proc = [get_dict_value(dict_tauyz, (x,z)) for x,z in zip(_x, _z)]
        tau_yy_proc = [get_dict_value(dict_tauyy, (x,z)) for x,z in zip(_x, _z)]
        # np array
        data_tau = np.zeros([nglob, 3], dtype=ff)
        data_tau[:,0] = np.array(tau_xy_proc)
        data_tau[:,1] = np.array(tau_yz_proc)    
        data_tau[:,2] = np.array(tau_yy_proc)    
        #
        x_debug.append(_x)
        z_debug.append(_z)
        xy_debug.append(data_tau[:,0])
        yz_debug.append(data_tau[:,1])
        yy_debug.append(data_tau[:,2])
        if info: print ('Number of points = ', nglob)   
        if info: print ('Data size = ', len(_x))   
        if (len(_x) != len(_z)) or (len(_x) != len(tau_xy_proc)) or  (len(_x) != len(tau_yz_proc)):
            print ('PROBLEM!!!')
            print ('Grid points of processor do not match interpolated value array')
            break
        print ('Writing ...')
        if write: 
            numRows = nglob
            numRowArr = np.array([numRows], dtype=ff)
            # I - number of points, nglob
            fileObj = open(fout, 'wb')
            numRowArr.tofile(fileObj)
            # II - tau_xy and tau_yz, and tau_yy
            for i in range(numRows):
                lineArr = data_tau[i,:]
                lineArr.tofile(fileObj)    
            fileObj.close()
            print ('done!')    
        print ()
    print('***')
    return 
###


def read_STF_file(filename='',NGLOB=None,NT=None):
    import numpy as np  
    
    try :
        with open(filename, 'rb') as fid:
            data_array = np.fromfile(fid,np.float32)
    except : 
        print('Velocity file does not exist')
    #
    print ('length of read array: ', len(data_array))
    print ('expected length (nglob*nt): ', NGLOB*NT)        
    data = np.zeros((NGLOB,NT))
    for it in range(NT):
        index1 = it* NGLOB
        index2 = (it+1)*NGLOB
        data[:,it] = data_array[index1:index2]
    #
    print ('*')
    return data
### 


def check_repeating_fault_nodes(self):
    print ('check_repeating_fault_nodes ...')
    # remove repeating nodes
    nx, nz = len( set(list(self.fault['x'])) ), len( set(list(self.fault['z'])) )

    print ('Non-repeating dimensions (nx,nz): ', nx, nz)
    print ('nx*nz: ', nx*nz)
    counted,repeated,index_2keep,index_repeateds = [],[],[],[]
    for ii, (_x, _z) in enumerate( zip(self.fault['x'], self.fault['z']) ):
        if (_x,_z) in counted: 
            repeated.append((_x, _z) )
            index_repeateds.append(ii)
        else:
            counted.append( (_x, _z) )
            index_2keep.append(ii)
    ##
    print ('uniques, repeated, uniques: ',len(counted), len(repeated), len(counted)+len(repeated))
    return nx,nz,index_2keep
###


def get_STF(rep='',nproc=1,info=True):
    '''Reading stf file per proc and assembles them'''
    import numpy as np

    STF_per_proc = []
    for iproc in range(nproc):
        filename = rep+'proc_'+'%05i' % (iproc)+ '_STF.dat'
        if info: print ('iproc, filename: ', iproc, filename)
        try:
            STF_per_proc.append(np.genfromtxt(filename, usecols=0))
        except:
            pass
        if info: print ('*')
    ##
    STF_all = np.sum ( np.array(STF_per_proc), axis=0 )
    print ('Shape of STF_all: ', STF_all.shape)
    print ('Done!')
    return STF_all
###


# def plot_STF_and_spectrum(self, jump_ko=10,\
#                 pow_ko=20.0,fmax=10.0, lims=[-0.1,20.0], \
#                 is_smooth=True,is_padding=False, rep_store='./STORE/', is_plot=True):
#     ''' computes STF and its spectrum. requires STF and dt. compute spectrum 
#     with Brune\'s and Ji and Archuleta's spectra.'''
#     from scipy import fft
#     sys.path.append('./')
#     from Class_SPECFEM3D import get_FFT,ko


#     # use padding only for spectrum calculation
#     dt = self.dt
#     self.time = np.arange(len(self.STF))* dt
#     if is_padding:
#         time = np.arange(20*len(self.STF))* dt
#         STF = np.zeros(time.shape)
#         STF[:len(self.STF)] = self.STF
#     else:
#         STF = self.STF
#         time = np.arange(len(self.STF))* dt
#     #
#     print ('Computing spectrum ...')
#     # Spectrum  
#     df = 1.0/ dt
#     N = len(STF)
#     _f = np.linspace(0.0, 1.0/(2.0*dt), N//2)
#     spec = abs(fft.fft(STF))* dt 
#     freq = _f[1:]
#     self.spec_moment_rate = spec[:int(N/2)-1]  
#     self.freq_source = freq
#     self.M0 = self.spec_moment_rate[0]
#     self.Mw = (np.log10(self.M0)- 9.1)/ 1.5
#     print('Mw: ', self.Mw)    
#     #
#     if is_smooth:
#         print ('Smoothening by konno-ohmachi ...')
#         cdt = (freq <= fmax)
#         f_sm = freq[cdt][::jump_ko]
#         self.spec_moment_rate_sm = ko(self.spec_moment_rate[cdt][::jump_ko], df, pow_ko)      
#         self.freq_source_sm = f_sm        
#     # brune's and ji & archuleta
#     fc1, fc2, spec_JA19 = get_JA19_spectrum(Mw=self.Mw, f=self.freq_source)
#     self.specD_ja = self.M0* spec_JA19
#     fc = (fc1* fc2)** 0.5
#     print ('fc (geometric mean): ', '%.2f' %(fc) )                             
#     self.brune = self.M0/ (1.0+(self.freq_source/ fc)**2)
#     #
#     print ('Pickling ...')
#     try: os.makedirs(rep_store)
#     except: pass
#     np.save(rep_store+'/STF.npy', self.STF)    
#     np.save(rep_store+'/time.npy', self.time)    
#     np.save(rep_store+'/freq_source.npy', self.freq_source)
#     np.save(rep_store+'/spec_brune.npy', self.brune)
#     np.save(rep_store+'/spec_JiArchuleta.npy', self.specD_ja)
#     np.save(rep_store+'/spec_moment_rate.npy', self.spec_moment_rate)
#     try: np.save(rep_store+'/spec_moment_rate_sm.npy', self.spec_moment_rate_sm)
#     except: pass
#     try: np.save(rep_store+'/freq_source_sm.npy', self.freq_source_sm)
#     except: pass    
#     print ('*')    
#     #
#     if is_plot:
#         plt.close('all')
#         print ('Plotting ...')
#         plt.figure(figsize=(8,4))
#         plt.subplot(121)
#         plt.xlim(lims[0],lims[1])
#         plt.plot(self.time, self.STF,'k',lw=1)
#         plt.grid() 
#         plt.xlabel('Time (s)'); plt.ylabel('Moment rate (Nm/s)')
#         #
#         plt.subplot(122)
#         plt.xlim(1e-3,fmax)
#         try: plt.loglog(self.freq_source_sm, self.spec_moment_rate_sm,'k', lw=1)    
#         except: plt.loglog(self.freq_source, self.spec_moment_rate, 'k', lw=1)
#         plt.loglog(self.freq_source, self.brune, c='gray', lw=1.0, linestyle='-.', label='w2 model')
#         plt.loglog(self.freq_source, self.specD_ja, c='gray', lw=1.0, linestyle=':', label='JA19_2S')
#         plt.axvline(x=fc1, c='pink', linestyle='-',lw=2, alpha=0.5)
#         plt.axvline(x=fc2, c='pink', linestyle='-',lw=2, alpha=0.5)
#         plt.xlabel('Frequency (Hz)')
#         plt.grid(True, which="both", ls=":", alpha=0.5)
#         plt.tight_layout()
#         plt.show()
#     return self
# ###