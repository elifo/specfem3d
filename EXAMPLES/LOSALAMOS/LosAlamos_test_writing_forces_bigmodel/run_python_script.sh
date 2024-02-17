# DONT DO ANYTHING WITHOUT THIS STEP!
module load python3/3.8.5

# SWITCH TO VIRTUAL ENV
source  /central/groups/enceladus/ELIF/python-environments/my-test-venv/bin/activate

# RESEVRE NODES FOR MULTIPROCS
srun --pty -t 2:00:00  -n 4 /bin/bash -l

# RUN SCRIPT


# to deactivate cancel srun
