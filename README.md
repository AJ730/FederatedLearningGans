# DistributedGAN
DistributedFederatedGAN


Project for research project.

This uses tensorflow, pytorch, mpi4py and Ray. This project is an adaptation of https://arxiv.org/abs/1811.03850

Implementation of unique architure with FedAvg and MDGan alg:

![architecture](./architecture/FLGAN.png)
    
![architecture](./architecture/MDGAN.png)

usage:

    - run pip install pytorch_requirements.txt
    - run main file in DistributedGanPytorch
    
                            or
    
    - run pip install tensorflow_requirements.txt
    - run main file in DistributedGanTensorflow


**Final Result for 1 discriminator and 1 generator with Federated Leanring server**
   ![architecture](./results/batch12.png) 
