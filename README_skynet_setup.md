UniVL Running on Skynet Walkthrough

Video walkthrough available here: https://youtu.be/83mumT-N0D4

Execute the following in order. Be sure to do so from your personal directory in Skynet, not the shared directory.

srun --gres gpu:4 -p debug -J "Job Name" --pty bash

git clone https://github.com/J-Himes/sthvl
cd sthvl

./bin/skynet/skynet_setup.sh
./bin/skynet/test_skynet.sh

Run this command and copy the ID for the job you previously created once it appears:
squeue -u <your_username>

Run this command using the job ID you copied to cancel the job you were using:
scancel <JOBID>
