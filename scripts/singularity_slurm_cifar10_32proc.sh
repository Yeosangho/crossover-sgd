#!/bin/sh
#SBATCH -J singularity
#SBATCH --gres=gpu:2

#SBATCH --time=48:00:00
#SBATCH -p cas_v100_2
#SBATCH --comment pytorch
#SBATCH -N 2 # number of node
#SBATCH -n 32
#SBATCH –o %x_%j.out
#SBATCH -e %x_%j.err
echo $SLURM_JOBID
cat singularity_slurm_cr_128k_14.5.sh
export PATH=$PATH:/apps/applications/singularity/3.1.0/bin
module purge
module load gcc/8.3.0 cuda/10.0 cudampi/openmpi-3.1.0
module load singularity/3.1.0
#module load gcc/8.3.0 cuda/10.0 cudampi/openmpi-3.1.0 conda/pytorch_1.0
export PROC_PER_NODE=16
export PROCESS_NUM=32
export GROUP_NUM=8
mpirun -n $PROCESS_NUM  -npernode $PROC_PER_NODE singularity exec --nv /scratch/x2026a02/200909_pytorch.sif /opt/anaconda3/envs/pytorch4/bin/python /scratch/x2026a02/cifar10_210303.py --batch_size 256 --world_size $PROCESS_NUM --crossover False --tag $SLURM_JOBID --lars True --allreduce False --baselr 3.2 --local_itr 1 --clip_grad False --sync_grad False --sync_lars_start_epoch 202 --sync_lars_group_size $PROCESS_NUM --chromosome fine --manual_seed 776 --gpu_per_node 2 --proc_per_node $PROC_PER_NODE --groupnum $GROUP_NUM
#srun python /scratch/x2026a02/test_pytorch.py
