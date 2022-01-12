# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

"""# Import Libraries"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils.model import select_model
from utils.options import parse_args_function
from utils.dataset import Dataset



def handle_stage(stage, model):
    #print("------------------------------------------")
    #print(model.module)
    if args.stage == 0:
        lambda_1 = 0.1
        lambda_2 = 1
        lambda_3 = 300
    elif args.stage == 1: # train only the resnet and graphnet
        lambda_1 = 0.01
        lambda_2 = 0
        lambda_3 = 0
        
    elif args.stage == 2: #fix resnet and graphnet and train graphunet
        lambda_1 = 0
        lambda_2 = 1
        lambda_3 = 0
        disable_params(model.module.resnet.parameters())
        disable_params(model.module.graphnet.parameters())
    else:# fix resnet, graphnet and graphunet and train the mesh generators
        lambda_1 = 0
        lambda_2 = 0
        lambda_3 = 300
        disable_params(model.module.resnet.parameters())
        disable_params(model.module.graphnet.parameters())
        disable_params(model.module.graphunet.parameters())

    return lambda_1, lambda_2, lambda_3


def disable_params(params):
    for param in params:
        param.requires_grad = False


def infer_calc_loss(inputs, labels2d, labels3d, model, criterion, lambda_1, lambda_2, lambda_3, generate_hand_mesh=False, mesh=None):
    """ Running Inference and Calculate Loss function """    

    #outputs2d_init, outputs2d, outputs3d, outputMesh = model(inputs)
    outputs2d_init, outputs2d, outputs3d, output_mesh3d = model(inputs)

    # Calculate loss for hand and object separately for each component (resnet, graphnet and graphunet)
    loss2d_init = criterion(outputs2d_init, labels2d)
    loss2d = criterion(outputs2d, labels2d)
    loss3d = criterion(outputs3d, labels3d)
    loss = (lambda_1)*loss2d_init + (lambda_1)*loss2d + (lambda_2)*loss3d

    if (generate_hand_mesh):
        loss3d_hand_mesh = criterion(output_mesh3d, mesh)
        loss3d_mesh = (lambda_3) * loss3d_hand_mesh
        loss += loss3d_mesh

    return outputs2d_init, outputs2d, outputs3d, output_mesh3d, loss


args = parse_args_function()

"""# Load Dataset"""

root = args.input_file

print('wwwwwww',root)

#mean = np.array([120.46480086, 107.89070987, 103.00262132])
#std = np.array([5.9113948 , 5.22646725, 5.47829601])

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])

if args.train:
    trainset = Dataset(root=root, load_set='train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    print('Train files loaded')

if args.val:
    valset = Dataset(root=root, load_set='val', transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=6)
    
    print('Validation files loaded')

if args.test:
    testset = Dataset(root=root, load_set='test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=6)
    
    print('Test files loaded')

"""# Model"""

use_cuda = False
if args.gpu:
    print("Model Training on GPU",args.gpu_number[0])
    use_cuda = True

model = select_model(args.model_def)
#print(model.resnet)


if use_cuda and torch.cuda.is_available():
    print("second condition is true")
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=args.gpu_number)

"""# Load Snapshot"""

if args.pretrained_model != '':
    model.load_state_dict(torch.load(args.pretrained_model))
    losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
    start = len(losses)
else:
    losses = []
    start = 0

"""# Optimizer"""

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
scheduler.last_epoch = start


lambda_1, lambda_2, lambda_3 = handle_stage(args.stage, model)

"""# Train"""

if args.train:
    print('Begin training the network...')
    
    for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
        print(epoch)
        running_loss = 0.0
        train_loss = 0.0
        for i, tr_data in enumerate(trainloader):
            # get the inputs
            inputs, labels2d, labels3d, labelMesh = tr_data    
            # wrap them in Variable
            inputs = Variable(inputs)
            labels2d = Variable(labels2d)
            labels3d = Variable(labels3d)
            labelMesh = Variable(labelMesh)
            
            if use_cuda and torch.cuda.is_available():
                inputs = inputs.float().cuda(device=args.gpu_number[0])
                labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                labels3d = labels3d.float().cuda(device=args.gpu_number[0])
                labelMesh = labelMesh.float().cuda(device=args.gpu_number[0])
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            _, _, _, _, loss = infer_calc_loss(inputs, labels2d, labels3d, model, criterion, lambda_1, lambda_2, lambda_3, generate_hand_mesh=True, mesh=labelMesh)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.data
            train_loss += loss.data
            if (i+1) % args.log_batch == 0:    # print every log_iter mini-batches
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / args.log_batch))
                running_loss = 0.0
                
        if args.val and (epoch+1) % args.val_epoch == 0:
            val_loss = 0.0
            for v, val_data in enumerate(valloader):
                # get the inputs
                inputs, labels2d, labels3d, labelMesh = val_data
                
                # wrap them in Variable
                inputs = Variable(inputs)
                labels2d = Variable(labels2d)
                labels3d = Variable(labels3d)
                labelMesh = Variable(labelMesh)
        
                if use_cuda and torch.cuda.is_available():
                    inputs = inputs.float().cuda(device=args.gpu_number[0])
                    labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                    labels3d = labels3d.float().cuda(device=args.gpu_number[0])
                    labelMesh = labelMesh.float().cuda(device=args.gpu_number[0])
        
                _, _, _, _, loss = infer_calc_loss(inputs, labels2d, labels3d, model, criterion, lambda_1, lambda_2, lambda_3, generate_hand_mesh=True, mesh=labelMesh)
                val_loss += loss.data
            print('val error: %.5f' % (val_loss / (v+1)))
        losses.append((train_loss / (i+1)).cpu().numpy())
        
        if (epoch+1) % args.snapshot_epoch == 0:
            torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
            np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))

        # Decay Learning Rate
        scheduler.step()
    
    print('Finished Training')

"""# Test"""

if args.test:
    print('Begin testing the network...')
    
    running_loss = 0.0
    for i, ts_data in enumerate(testloader):
        # get the inputs
        inputs, labels2d, labels3d = ts_data
        
        # wrap them in Variable
        inputs = Variable(inputs)
        labels2d = Variable(labels2d)
        labels3d = Variable(labels3d)

        if use_cuda and torch.cuda.is_available():
            inputs = inputs.float().cuda(device=args.gpu_number[0])
            labels2d = labels2d.float().cuda(device=args.gpu_number[0])
            labels3d = labels3d.float().cuda(device=args.gpu_number[0])

        outputs2d_init, outputs2d, outputs3d = model(inputs)
        
        loss = criterion(outputs3d, labels3d)
        running_loss += loss.data
    print('test error: %.5f' % (running_loss / (i+1)))
