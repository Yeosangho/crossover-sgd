#!/bin/sh
#SBATCH -J singularity
#SBATCH --gres=gpu:2

#SBATCH --time=1:00:00
#SBATCH -p ivy_v100_2
#SBATCH --comment pytorch
#SBATCH -N 1 # number of node
#SBATCH -n 8
#SBATCH –o %x_%j.out
#SBATCH -e %x_%j.err
echo $SLURM_JOBID
cat singularity_slurm_sgp1.sh
export PATH=$PATH:/apps/applications/singularity/3.1.0/bin
module purge
module load gcc/8.3.0 cuda/10.0 cudampi/openmpi-3.1.0
module load singularity/3.1.0
#module load gcc/8.3.0 cuda/10.0 cudampi/openmpi-3.1.0 conda/pytorch_1.0

mpirun -n 8  -npernode 8 singularity exec --nv /scratch/x2026a02/pytorch_201114.sif /opt/anaconda3/envs/pytorch4/bin/python /scratch/x2026a02/imagenet_lmdb_test.py --batch_size 32 --world_size 8 --crossover False --tag $SLURM_JOBID --lars True --allreduce False --amp True --lrdecay poly --lars_coef 0.0025 --baselr 9 --warmup_epoch 36 --local_itr 21 --clip_grad True --sync_grad False --sync_lars_start_epoch 200 --sync_lars_group_size 8 --chromosome fine --maxepoch 90 --manual_seed 776 --gpu_per_node 2
#srun python /scratch/x2026a02/test_pytorch.py
