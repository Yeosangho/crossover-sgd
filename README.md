# Crossover-SGD :  A gossip-based communication in distributed deep learning for alleviating large mini-batch problem and enhancing scalability

---------------
## Evaluation Results
---------------


   <img src="images/eval5.PNG" width="100%" height="100%" title="px(픽셀) 크기 설정" alt="RubberDuck"></img>

  <img src="images/eval6.PNG" width="100%" height="100%" title="px(픽셀) 크기 설정" alt="RubberDuck"></img>  


  <img src="images/eval3.PNG" width="100%" height="100%" title="px(픽셀) 크기 설정" alt="RubberDuck"></img>

  <img src="images/eval4.PNG" width="100%" height="100%" title="px(픽셀) 크기 설정" alt="RubberDuck"></img>


---------------
## Execution Commands
-------------------

```
mpirun -n 32  -npernode 4 singularity exec --nv /scratch/x2026a02/200909_pytorch.sif /opt/anaconda3/envs/pytorch4/bin/python /scratch/x2026a02/imagenet_test.py --batch_size 128 --world_size 32 --crossover True --tag $SLURM_JOBID --lars True --allreduce False --amp True --lrdecay poly --lars_coef 0.0025 --baselr 11 --warmup_epoch 36 --local_itr 21 --clip_grad True --sync_grad False --sync_lars_start_epoch 200 --sync_lars_group_size 8 --chromosome fine --maxepoch 90 --manual_seed 776 --gpu_per_node 2

```

* modified version (slurm, nccl backend) 

```
srun /opt/anaconda3/envs/pytorch4/bin/python /scratch/x2026a02/imagenet_test.py --batch_size 128 --world_size 32 --crossover True --tag $SLURM_JOBID --lars True --allreduce False --amp True --lrdecay poly --lars_coef 0.0025 --baselr 11 --warmup_epoch 36 --local_itr 21 --clip_grad True --sync_grad False --sync_lars_start_epoch 200 --sync_lars_group_size 8 --chromosome fine --maxepoch 90 --manual_seed 776 --gpu_per_node 2

```

