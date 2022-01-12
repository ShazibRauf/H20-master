# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.graphunet import GraphUNet, GraphNet, MeshGen
from models.resnet import resnet50, resnet10, resnet18, resnet152


class HopeNet(nn.Module):

    def __init__(self, generate_hand_mesh=False, generate_obj_mesh=False,):
        super(HopeNet, self).__init__()
        self.resnet = resnet50(pretrained=False, num_classes=50*2)
        self.graphnet = GraphNet(in_features=2050, out_features=2)
        self.graphunet = GraphUNet(in_features=2, out_features=3)

        self.generate_hand_mesh = generate_hand_mesh
        self.generate_obj_mesh = generate_obj_mesh
        
        if self.generate_hand_mesh:
            print("Adding hand generator .. ")
            self.handMeshgen_left = MeshGen(in_features=2053, out_features=3, selectHand='left')
            self.handMeshgen_right = MeshGen(in_features=2053, out_features=3, selectHand='right')
            if self.generate_obj_mesh:
                print("Adding object generator .. ")
                self.objectmeshgen = MeshGen(in_features=2053, out_features=3, n=1000)



    def forward(self, x):
        #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        points2D_init, features = self.resnet(x)
        #print(points2D_init.shape)
        #print(features.shape)
        #print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
        features = features.unsqueeze(1).repeat(1, 50, 1)
        # batch = points2D.shape[0]
        in_features = torch.cat([points2D_init, features], dim=2)
        points2D = self.graphnet(in_features)
        #print("llllllllllllllllllllllllllllllllllllllllllllllllllllllllllll")
        points3D = self.graphunet(points2D)
        #print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
        mesh = None
        
        if self.generate_hand_mesh:
            in_features = torch.cat([points2D, points3D, features], dim=2)
            #print(in_features.shape)
            handmesh_left = self.handMeshgen_left(in_features)
            #print("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
            handmesh_right = self.handMeshgen_right(in_features)
            if self.generate_obj_mesh:
                objectmesh = self.objectmeshgen(in_features)
                mesh = torch.cat([handmesh_left, handmesh_right, objectmesh], dim=1)
            else:
                mesh = torch.cat([handmesh_left, handmesh_right], dim=1)



        return points2D_init, points2D, points3D, mesh