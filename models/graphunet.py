import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse

class GraphConv(nn.Module):
    
    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        #self.adj_sq = adj_sq
        self.activation = activation
        #self.scale_identity = scale_identity
        #self.I = Parameter(torch.eye(number_of_nodes, requires_grad=False).unsqueeze(0))


    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L
    
    
    def laplacian_batch(self, A_hat):
        #batch, N = A.shape[:2]
        #if self.adj_sq:
        #    A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
        #I = torch.eye(N).unsqueeze(0).to(device)
        #I = self.I
        #if self.scale_identity:
        #    I = 2 * I  # increase weight of self connections
        #A_hat = A + I
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L


    def forward(self, X, A):
        batch = X.size(0)
        #A = self.laplacian(A)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        #X = torch.bmm(A_hat, X)
        #print("--->",self.laplacian_batch(A_hat).size())
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X


class GraphPool(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(GraphPool, self).__init__()
        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)


    def forward(self, X):
        X = X.transpose(1, 2)
        X = self.fc(X)
        X = X.transpose(1, 2)
        return X


class GraphUnpool(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(GraphUnpool, self).__init__()
        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)


    def forward(self, X):
        X = X.transpose(1, 2)
        X = self.fc(X)
        X = X.transpose(1, 2)
        return X


class GraphUNet(nn.Module):

    def __init__(self, in_features=2, out_features=3):
        super(GraphUNet, self).__init__()

        self.A_0 = Parameter(torch.eye(50).float().cuda(), requires_grad=True)
        self.A_1 = Parameter(torch.eye(25).float().cuda(), requires_grad=True)
        self.A_2 = Parameter(torch.eye(12).float().cuda(), requires_grad=True)
        self.A_3 = Parameter(torch.eye(6).float().cuda(), requires_grad=True)
        self.A_4 = Parameter(torch.eye(3).float().cuda(), requires_grad=True)
        self.A_5 = Parameter(torch.eye(1).float().cuda(), requires_grad=True)

        self.gconv1 = GraphConv(in_features, 4)  # 29 = 21 H + 8 O
        self.pool1 = GraphPool(50, 25)

        self.gconv2 = GraphConv(4, 8)  # 15 = 11 H + 4 O
        self.pool2 = GraphPool(25, 12)

        self.gconv3 = GraphConv(8, 16)  # 7 = 5 H + 2 O
        self.pool3 = GraphPool(12, 6)

        self.gconv4 = GraphConv(16, 32)  # 4 = 3 H + 1 O
        self.pool4 = GraphPool(6, 3)

        self.gconv5 = GraphConv(32, 64)  # 2 = 1 H + 1 O
        self.pool5 = GraphPool(3, 1)

        self.fc1 = nn.Linear(64, 20)

        self.fc2 = nn.Linear(20, 64)

        self.unpool6 = GraphUnpool(1, 3)
        self.gconv6 = GraphConv(128, 32)

        self.unpool7 = GraphUnpool(3, 6)
        self.gconv7 = GraphConv(64, 16)

        self.unpool8 = GraphUnpool(6, 12)
        self.gconv8 = GraphConv(32, 8)

        self.unpool9 = GraphUnpool(12, 25)
        self.gconv9 = GraphConv(16, 4)

        self.unpool10 = GraphUnpool(25, 50)
        self.gconv10 = GraphConv(8, out_features, activation=None)

        self.ReLU = nn.ReLU()

    def _get_decoder_input(self, X_e, X_d):
        return torch.cat((X_e, X_d), 2)

    def forward(self, X):
        #print("Start Graph U net")
        X_0 = self.gconv1(X, self.A_0)
        X_1 = self.pool1(X_0)

        X_1 = self.gconv2(X_1, self.A_1)
        X_2 = self.pool2(X_1)

        X_2 = self.gconv3(X_2, self.A_2)
        X_3 = self.pool3(X_2)

        X_3 = self.gconv4(X_3, self.A_3)
        X_4 = self.pool4(X_3)

        X_4 = self.gconv5(X_4, self.A_4)
        X_5 = self.pool5(X_4)

        global_features = self.ReLU(self.fc1(X_5))
        global_features = self.ReLU(self.fc2(global_features))

        X_6 = self.unpool6(global_features)
        X_6 = self.gconv6(self._get_decoder_input(X_4, X_6), self.A_4)

        X_7 = self.unpool7(X_6)
        X_7 = self.gconv7(self._get_decoder_input(X_3, X_7), self.A_3)

        X_8 = self.unpool8(X_7)
        X_8 = self.gconv8(self._get_decoder_input(X_2, X_8), self.A_2)

        X_9 = self.unpool9(X_8)
        X_9 = self.gconv9(self._get_decoder_input(X_1, X_9), self.A_1)

        X_10 = self.unpool10(X_9)
        X_10 = self.gconv10(self._get_decoder_input(X_0, X_10), self.A_0)

        #print("End graph u net")
        return X_10


class GraphNet(nn.Module):
    
    def __init__(self, in_features=2, out_features=2):
        super(GraphNet, self).__init__()

        self.A_hat = Parameter(torch.eye(50).float().cuda(), requires_grad=True)
        
        self.gconv1 = GraphConv(in_features, 128)
        self.gconv2 = GraphConv(128, 16)
        self.gconv3 = GraphConv(16, out_features, activation=None)
        #print("out features: ", out_features)
        
    
    def forward(self, X):
        #print("start")
        #print(X.shape)
        #print(self.A_hat.shape)
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        X_2 = self.gconv3(X_1, self.A_hat)
        #print("end")
        return X_2



class MeshGen(nn.Module):
    def __init__(self, in_features=3, out_features=3, n=778, fixed_matrix=True, selectHand='left'):
        super(MeshGen, self).__init__()

        l3 = n
        l2 = l3 // 4
        l1 = l2 // 4
        
        self.A_1 = Parameter(torch.eye(l1).float().cuda(), requires_grad=True)
        self.A_2 = Parameter(torch.eye(l2).float().cuda(), requires_grad=True)
        self.A_3 = Parameter(torch.eye(l3).float().cuda(), requires_grad=True)

        if fixed_matrix:
            print("Loading Fixed Adjacency Matrix ..")
                
            if n == 778: # Hand mesh generator
                if (selectHand == 'left'):
                    A = torch.from_numpy(scipy.sparse.load_npz('/home/satti/H20-master-main/handAdjacency_lefthand.npz').toarray())
                else:
                    A = torch.from_numpy(scipy.sparse.load_npz('/home/satti/H20-master-main/handAdjacency_righthand.npz').toarray())
            elif n == 1778: # Hand-Object mesh generator
                A = torch.from_numpy(scipy.sparse.load_npz('handObjectAdjacency.npz').toarray())    
            else: # Object mesh generator
                A = torch.from_numpy(scipy.sparse.load_npz('objectAdjacency.npz').toarray())
            
            self.A_3 = Parameter(A.float().cuda(), requires_grad=False)
            
            # else: # TODO: object or hand-object mesh generator

        """ Mesh generator, associated loss factor 300"""


        self.unpool1 = GraphUnpool(50, l1)
        self.gconv11 = GraphConv(in_features, 512) # From x features per node to y
        self.gconv12 = GraphConv(512, 256)  # from x features to y features per node
        
        self.unpool2 = GraphUnpool(l1, l2) # From x nodes to y node
        self.gconv21 = GraphConv(256, 128)
        self.gconv22 = GraphConv(128, 64) 
        
        self.unpool3 = GraphUnpool(l2, l3) # From x nodes to y nodes
        self.gconv31 = GraphConv(64, 16)
        self.gconv32 = GraphConv(16, out_features, activation=None)
        
        
    def forward(self, X):


        X = self.unpool1(X)
        X = self.gconv11(X, self.A_1)
        X = self.gconv12(X, self.A_1)
        
        X = self.unpool2(X)
        X = self.gconv21(X, self.A_2)
        X = self.gconv22(X, self.A_2)

        X = self.unpool3(X)
        X = self.gconv31(X, self.A_3)
        X = self.gconv32(X, self.A_3)


        return X
