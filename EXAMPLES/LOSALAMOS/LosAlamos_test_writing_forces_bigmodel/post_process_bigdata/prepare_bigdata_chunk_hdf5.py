import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import time
import os


directory = 'OUTPUT_CHUNKS_BENCHMARKMODEL_FORCES'
if not os.path.exists(directory):
    os.makedirs(directory)
#


## TO-DO
# change station name list
# write only chunk stations
# check out comp. time
# check out signals


time_start = time.process_time()


rep = '../OUTPUT_FILES/'
is_velocity = True


df_stations = pd.read_csv(rep+'output_list_stations.txt',header=None,delim_whitespace=True)
df_stations.columns = ['Station','net','x','y','z']
NSTA = len(df_stations)
print ('Station number (NSTA): ', NSTA)


CHUNKSIZE = 10000
NCHUNK = int(NSTA/ CHUNKSIZE)+ 1
print ('NCHUNK', NCHUNK)

NT = 10000 # dt_sample=5e-3; dt_simulation=dt_sample/jump  #only used for print info
suffix = '.sema'
prefix = 'EL.LOS'
sep = '_'



###
from multiprocessing import Pool
def reader(filename):
    jump = 1
    return pd.read_csv(filename,header=None,delim_whitespace=True,usecols=[1])[::jump]
def read_given_files(file_list):
    pool = Pool(4) # number of cores you want to use
    df_list = pool.map(reader, file_list) #creates a list of the loaded df's
    print ('size of df_list: ', len(df_list))
    print ('size of df_list[0]: ', len(df_list[0]))
    df_test = pd.concat(df_list) # concatenates all the df's into a single df
    return df_test
###

for ichunk in range(NCHUNK):
    print ('ichunk: ', ichunk)
    # new file per chunk
    filename = directory+ '/LosAlamos_ichunk_'+str(ichunk)+'.h5'
    # later only write the chunk statiions' coordinates
    #df_coord.to_hdf(filename, key='coords', mode='w')

    # choose the stations in the chunk
    ibeg, iend = ichunk*CHUNKSIZE, min((ichunk+1)*CHUNKSIZE, NSTA)
    df_station_ichunk = df_stations.iloc[ibeg:iend]
    df_station_ichunk.to_hdf(filename, key='Stations', mode='w')
    for compo in  ['X', 'Y', 'Z'] :
        print ('compo: ', compo)
        file_list = []
        #for ista in range(ibeg, iend):
        for sta, net in zip(df_station_ichunk['Station'].values, df_station_ichunk['net'].values):
            #fname = rep+ prefix+'%06d'% (ista)+'.FX'+compo+suffix # to revise & use with large stat nb 
            fname = rep+ net+'.'+sta+'.FX'+compo+suffix # to revise & use with large stat nb 
            file_list.append(fname)
        ##
        print ('len(file_list): ', len(file_list))
        # read the chunk
        df_ichunk_icompo = read_given_files(file_list)
        # write out 
        df_ichunk_icompo.to_hdf(filename, key='V'+sep+compo) 
        #print ('df size, NSTA_chunk : ', df_ichunk_icompo.size, int( df_ichunk_icompo.size/NT))
        print ('df size (NSTA_chunk*NT): ', df_ichunk_icompo.size)
    ##
    print ('*')
    time_elapsed = (time.process_time() - time_start)
    print ('time_elapsed (min): ', time_elapsed/60.0)
##
print ('***')
##
