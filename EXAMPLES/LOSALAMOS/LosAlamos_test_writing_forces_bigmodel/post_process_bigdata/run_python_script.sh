# DONT DO ANYTHING WITHOUT THIS STEP!
module load python3/3.8.5

# SWITCH TO VIRTUAL ENV
source  /central/groups/enceladus/ELIF/python-environments/my-test-venv/bin/activate

# RESEVRE NODES FOR MULTIPROCS
srun --pty -t 12:00:00  -n 2  /bin/bash -l

# RUN SCRIPT
python test_bigdata_chunk_hdf5.py  > log.txt &

# to deactivate cancel srun
