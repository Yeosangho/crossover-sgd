#!/bin/sh
#SBATCH -J singularity
#SBATCH --gres=gpu:2

#SBATCH --time=48:00:00
#SBATCH -p cas_v100_2
#SBATCH --comment pytorch
#SBATCH -N 1 # number of node
#SBATCH -n 1
#SBATCH –o %x_%j.out
#SBATCH -e %x_%j.err
echo $SLURM_JOBID
cat singularity_slurm_cr_128k_14.5.sh
export PATH=$PATH:/apps/applications/singularity/3.1.0/bin
module purge
module load gcc/8.3.0 cuda/10.0 cudampi/openmpi-3.1.0
module load singularity/3.1.0
#module load gcc/8.3.0 cuda/10.0 cudampi/openmpi-3.1.0 conda/pytorch_1.0
export PROC_PER_NODE=1
export PROCESS_NUM=1
export GROUP_NUM=8
singularity exec --nv /scratch/x2026a02/200909_pytorch.sif /opt/anaconda3/envs/pytorch4/bin/python /scratch/x2026a02/cifar10_download.py