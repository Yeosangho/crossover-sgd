#!/bin/sh
#SBATCH -J singularity
#SBATCH --gres=gpu:2

#SBATCH --time=48:00:00
#SBATCH -p ivy_v100-32G_2
#SBATCH --comment pytorch
#SBATCH -N 4 # number of node
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
export PROC_PER_NODE=8
export PROCESS_NUM=32
mpirun -n $PROCESS_NUM  -npernode $PROC_PER_NODE singularity exec --nv /scratch/x2026a02/200909_pytorch.sif /opt/anaconda3/envs/pytorch4/bin/python /scratch/x2026a02/imagenet_test_3peer.py --batch_size 64 --world_size $PROCESS_NUM --crossover False --tag $SLURM_JOBID --lars True --allreduce False --amp True --lrdecay poly --lars_coef 0.0025 --baselr 9 --warmup_epoch 36 --local_itr 42 --clip_grad True --sync_grad False --sync_lars_start_epoch 200 --sync_lars_group_size $PROCESS_NUM --chromosome fine --maxepoch 90 --manual_seed 776 --gpu_per_node 2 --proc_per_node $PROC_PER_NODE
#srun python /scratch/x2026a02/test_pytorch.py
