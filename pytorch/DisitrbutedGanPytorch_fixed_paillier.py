# Deep Convolutional GANs

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from mpi4py import MPI
from torch import optim
from torch.autograd import Variable
from torch.utils.data import random_split
import torchvision.utils as vutils
from helperUtils import py_utils
import dill
import phe as paillier

MPI.pickle.__init__(dill.dumps, dill.loads)
""" 
We use dataLoader to get the images of the training set batch by batch.
We ust the shuffle = True because we want to get the dataset in random order so that we can train model more precisely.
We use num_worker = 5 which represent the number of thread and the worker servers to define the 
"""





# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Defining the copy of the generator to shuffle between diffrent severs
def copyGenerator():
    layer_num = 0
    for param in netG.parameters():
        # print(rank, "started")
        if (rank == 0):
            data = param.data.numpy().copy()
            # print(rank, data.shape)
        else:
            data = None
            # print(rank, data.shape)

        # print(rank, "before bcast")
        # comm.Barrier()
        data = comm.bcast(data, root=0)
        # print(rank, "after bcast")
        if (rank != 0):
            param.data = torch.from_numpy(data)
            print("Node rank " + str(rank) + " has synched generator layer " + str(layer_num))

        layer_num += 1
        # comm.Barrier()


# Peer2Peer shuffling of the Discriminator
def shuffleDiscriminators():
    if (rank != 0):
        layer_num = 0
        for param in netD.parameters():
            outdata = param.data.numpy().copy()
            indata = None

            if (rank != size - 1):
                comm.send(outdata, dest=rank + 1, tag=1)
            if (rank != 1):
                indata = comm.recv(source=rank - 1, tag=1)

            if (rank == size - 1):
                comm.send(outdata, dest=1, tag=2)
            if (rank == 1):
                indata = comm.recv(source=size - 1, tag=2)
            # Shuffling the Discriminator
            param.data = torch.from_numpy(indata)
            layer_num += 1


# Defining the generator

class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)


if __name__ == '__main__':
    #
    # # Creating the network to create the peer2peer connection for swaping of the Discriminator
    batchSize = 64  # We set the size of the batch.
    imageSize = 64  # We set the size of the generated images (64x64).
    clients = 3

    training_steps = 3
    # Creating the transformations
    transform = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    print(size)
    rank = comm.Get_rank()

    batch_size = 128

    dataset = None
    dataloader = None

    public_key = None
    private_key = None

    if (rank == 0):
        dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
        print("Files Downloaded", flush=True)

        partition1 = int(len(dataset) / 3)
        partition2 = int(len(dataset) / 3)
        partition3 = int(len(dataset)) - partition2 - partition1

        numpy_dataset_partition_per_client = random_split(dataset, [partition1, partition2, partition3])

        # recieving tag = 1
        for i in range(1, clients+1):
            comm.send(numpy_dataset_partition_per_client[i-1], i, tag=1)


    else:
        dataset = comm.recv(source=MPI.ANY_SOURCE, tag=1)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)

        util = py_utils.Util(dataset, dataloader)
        device = util.get_default_device()
        print("using:", device, flush=True)

    comm.barrier()
    print("synchronised", flush=True)


    # SEND PAILLIER KEY
    if (rank == 4):
        # recieving tag = 1
        pub,priv = paillier.generate_paillier_keypair(n_length=2048)

        for i in range(1, clients+1):
            comm.send([pub,priv], i, tag=1)


    if (rank!=4) and (rank!=0):
        keys = comm.recv(source=MPI.ANY_SOURCE, tag=1)
        public_key = keys[0]
        private_key = keys[1]
    


    comm.barrier()


    netG = G()
    netD = D()

    global_weights_generator = []
    global_weights_discriminator = []

    for epoch in range(3):

        if (rank == 0) and epoch == 0:
            global_weights_generator_init = weights_init
            global_weights_discriminator_init = weights_init
        else:
            global_weights_discriminator_init = None
            global_weights_generator_init = None

        global_weights_generator_init = comm.bcast(global_weights_generator_init, root=0)
        global_weights_discriminator_init = comm.bcast(global_weights_discriminator_init, root=0)
        #global_weights_generator = comm.bcast(global_weights_generator, root=0)
        #global_weights_discriminator = comm.bcast(global_weights_discriminator, root=0)

        print('generator_init', global_weights_generator_init, flush=True)
        print('generator',global_weights_generator, flush=True)

        comm.barrier()

        if rank != 0 and rank !=4:
            print("epoch ", epoch, flush=True)
            # Creating the generator

            if(epoch == 0):
                netG.apply(global_weights_generator_init)
                netD.apply(global_weights_discriminator_init)
            #else:
             #   netG.apply(global_weights_generator)
              #  netD.apply(global_weights_discriminator)

            criterion = nn.BCELoss()
            optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
            optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))


            assert (dataloader is not None)
            for i, data in enumerate(dataloader, 0):

                print("batch", i, flush=True)
                # 1st Step: Updating the weights of the neural network of the discriminator

                netD.zero_grad()

                print("training data", i, flush=True)
                real, _ = data
                input = Variable(real)
                target = Variable(torch.ones(input.size()[0]))
                output = netD(input)
                errD_real = criterion(output, target)

                # Training the discriminator with a fake image generated by the generator
                noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
                fake = netG(noise)
                target = Variable(torch.zeros(input.size()[0]))
                output = netD(fake.detach())
                errD_fake = criterion(output, target)

                # Backpropagating the total error
                errD = errD_real + errD_fake
                errD.backward()
                optimizerD.step()

                # 2nd Step: Updating the weights of the neural network of the generator

                netG.zero_grad()
                target = Variable(torch.ones(input.size()[0]))
                output = netD(fake)
                errG = criterion(output, target)
                errG.backward()
                optimizerG.step()

                # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps

                print(
                    '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
                    epoch, 25, i, len(dataloader), errD.item(), errG.item()),
                    flush=True)

                if i % 100 == 0:
                    vutils.save_image(real, '%s/real_samples.png' % "./results", normalize=True)
                    fake = netG(noise)
                    vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch),
                                      normalize=True)

        comm.barrier()
        for param in netG.parameters():
            print("here", flush=True)
            p = 0
            if(rank == 0 or rank == 4):
                p = 0
            else:
                p = public_key.encrypt(param.data/clients)
            global_weights_generator = comm.reduce(p, MPI.SUM, root=0)

            if (rank == 0):
                for i in range(1, clients+1):
                    comm.send(global_weights_generator, i, tag=1)

            if rank!=0 and rank!=4:
                param.data = private_key.decrypt(comm.recv(source=MPI.ANY_SOURCE, tag=1))
            comm.barrier()


        for param in netD.parameters():
            print("here2", flush=True)
            p = 0
            if(rank == 0 or rank == 4):
                p = 0
            else:
                p = public_key.encrypt(param.data/clients)
            global_weights_discriminator = comm.reduce(p,  MPI.SUM, root=0)

            if (rank == 0):
                for i in range(1, clients+1):
                    comm.send(global_weights_discriminator, i, tag=1)

            if rank!=0 and rank!=4:
                param.data = private_key.decrypt(comm.recv(source=MPI.ANY_SOURCE, tag=1))
            comm.barrier()



