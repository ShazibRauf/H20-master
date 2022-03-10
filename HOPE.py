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
    #print("----------Stage: "stage)
    #print(model.module)
    if args.stage == 0:
        lambda_1 = 0.1
        lambda_2 = 100
        lambda_3 = 100
        lambda_4 = 100
    elif args.stage == 1: # train only the resnet and graphnet
        lambda_1 = 0.01
        lambda_2 = 0
        lambda_3 = 0
        lambda_4 = 0
        
    elif args.stage == 2: #fix resnet and graphnet and train graphunet
        lambda_1 = 0
        lambda_2 = 1
        lambda_3 = 0
        lambda_4 = 0
        disable_params(model.module.resnet.parameters())
        disable_params(model.module.graphnet.parameters())
        disable_params(model.module.handMeshgen_left.parameters())
        disable_params(model.module.handMeshgen_right.parameters())
        disable_params(model.module.objectMeshgen.parameters())
    else:# fix resnet, graphnet and graphunet and train the mesh generators
        lambda_1 = 0
        lambda_2 = 0
        lambda_3 = 300
        lambda_4 = 0
        disable_params(model.module.resnet.parameters())
        disable_params(model.module.graphnet.parameters())
        disable_params(model.module.graphunet.parameters())
        disable_params(model.module.objectMeshgen.parameters())

    return lambda_1, lambda_2, lambda_3, lambda_4


def disable_params(params):
    for param in params:
        param.requires_grad = False


def infer_calc_loss(inputs, tsdf, labels2d, labels3d, model, criterion, lambda_1, lambda_2, lambda_3, lambda_4, generate_hand_mesh=False, generate_object_mesh=False, generate_combine_mesh=False, handMesh=None, objMesh=None, HandObjMesh=None):
    """ Running Inference and Calculate Loss function """    

    #outputs2d_init, outputs2d, outputs3d, outputMesh = model(inputs)
    #print("herrrrrrrrrrrrrrrrrrrrrrrrrrrr")
    outputs2d_init, outputs2d, outputs3d, output_handMesh, output_objMesh, outputs_combineMesh = model(inputs, tsdf)
    #print("done .......................................")
    #print("####################################333")
    #print(outputs_combineMesh.shape)

    # Calculate loss for hand and object separately for each component (resnet, graphnet and graphunet)
    loss2d_init = criterion(outputs2d_init, labels2d)
    loss2d = criterion(outputs2d, labels2d)
    loss3d = criterion(outputs3d, labels3d)
    loss = (lambda_1)*loss2d_init + (lambda_1)*loss2d + (lambda_2)*loss3d

    if (generate_hand_mesh):
        loss3d_hand_mesh = criterion(output_handMesh, handMesh)
        loss3d_mesh = (lambda_3) * loss3d_hand_mesh
        loss += loss3d_mesh


    if (generate_object_mesh):
        loss3d_obj_mesh = criterion(output_objMesh, objMesh)
        loss3d_mesh = (lambda_4) * loss3d_obj_mesh
        loss += loss3d_mesh

    #changes required
    if (generate_combine_mesh):
        loss3d_combine_mesh = criterion(outputs_combineMesh, HandObjMesh)
        loss3d_mesh = (lambda_4) * loss3d_combine_mesh
        loss += loss3d_mesh

    return outputs2d_init, outputs2d, outputs3d, output_handMesh, output_objMesh, outputs_combineMesh, loss


args = parse_args_function()

"""# Load Dataset"""

root = args.input_file

print('wwwwwww',root)

#mean = np.array([120.46480086, 107.89070987, 103.00262132])
#std = np.array([5.9113948 , 5.22646725, 5.47829601])

transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224, 224))])

if args.train:
    trainset = Dataset(root=root, load_set='train', transform=transform, depth = 1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32)
    
    print('Train files loaded')

if args.val:
    valset = Dataset(root=root, load_set='val', transform=transform, depth = 1)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=32)
    
    print('Validation files loaded')

if args.test:
    testset = Dataset(root=root, load_set='test', transform=transform, depth = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=32)
    
    print('Test files loaded')

"""# Model"""

use_cuda = False
if args.gpu:
    print("Model Training on GPU",args.gpu_number[0])
    use_cuda = True


model = select_model(args.model_def, depth=1, features3d=2048)
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


lambda_1, lambda_2, lambda_3, lambda_4 = handle_stage(args.stage, model)

"""# Train"""

if args.train:
    print('Begin training the network...')
    
    for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
        print(epoch)
        running_loss = 0.0
        train_loss = 0.0
        for i, tr_data in enumerate(trainloader):
            # get the inputs
            imgPath, inputs, tsdf, labels2d, labels3d, labelHandMesh, labelObjMesh, labelHandObjMesh = tr_data    
            # wrap them in Variable
            inputs = Variable(inputs)
            tsdf = Variable(tsdf)
            labels2d = Variable(labels2d)
            labels3d = Variable(labels3d)
            labelHandMesh = Variable(labelHandMesh)
            labelObjMesh = Variable(labelObjMesh)
            labelHandObjMesh = Variable(labelHandObjMesh)
            
            if use_cuda and torch.cuda.is_available():
                inputs = inputs.float().cuda(device=args.gpu_number[0])
                tsdf = tsdf.float().cuda(device=args.gpu_number[0])
                labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                labels3d = labels3d.float().cuda(device=args.gpu_number[0])
                labelHandMesh = labelHandMesh.float().cuda(device=args.gpu_number[0])
                labelObjMesh = labelObjMesh.float().cuda(device=args.gpu_number[0])
                labelHandObjMesh = labelHandObjMesh.float().cuda(device=args.gpu_number[0])
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            print("-------",type(inputs))
            _, _, _, _, _, _, loss = infer_calc_loss(inputs, tsdf, labels2d, labels3d, model, criterion, lambda_1, lambda_2, lambda_3, lambda_4, generate_hand_mesh=False, generate_object_mesh=False, generate_combine_mesh=True, handMesh=labelHandMesh, objMesh=labelObjMesh, HandObjMesh=labelHandObjMesh)
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
                imgPath, inputs, tsdf, labels2d, labels3d, labelHandMesh, labelObjMesh, labelHandObjMesh = val_data
                
                # wrap them in Variable
                inputs = Variable(inputs).float()
                tsdf = Variable(tsdf).float()
                labels2d = Variable(labels2d).float()
                labels3d = Variable(labels3d).float()
                labelHandMesh = Variable(labelHandMesh).float()
                labelObjMesh = Variable(labelObjMesh).float()
                labelHandObjMesh = Variable(labelHandObjMesh).float()
            
                if use_cuda and torch.cuda.is_available():
                    inputs = inputs.float().cuda(device=args.gpu_number[0])
                    tsdf = tsdf.float().cuda(device=args.gpu_number[0])
                    labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                    labels3d = labels3d.float().cuda(device=args.gpu_number[0])
                    labelHandMesh = labelHandMesh.float().cuda(device=args.gpu_number[0])
                    labelObjMesh = labelObjMesh.float().cuda(device=args.gpu_number[0])
                    labelHandObjMesh = labelHandObjMesh.float().cuda(device=args.gpu_number[0])
        
                _, _, _, _, _, _, loss = infer_calc_loss(inputs, tsdf, labels2d, labels3d, model, criterion, lambda_1, lambda_2, lambda_3, lambda_4, generate_hand_mesh=False, generate_object_mesh=False, generate_combine_mesh=True, handMesh=labelHandMesh, objMesh=labelObjMesh, HandObjMesh=labelHandObjMesh)
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
