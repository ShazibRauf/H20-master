# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
from PIL import Image
import torch
from manopth.manolayer import ManoLayer


from utils.tsdf_utils import run_tsdf, cubicSz
import cv2
import sys
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns


"""# Load Dataset"""




def calculate_center(point2d, point3d):    
    center2d = np.average(point2d[-8:], axis=0)
    center3d = np.average(point3d[-8:], axis=0)        
    
    return center2d, center3d



def load_mesh_from_manolayer(fullpose, beta, trans, handSelection):
    # mano_layer = ManoLayer(mano_root='../../../mano_v1_2/models', use_pca=False, ncomps=6, flat_hand_mean=True)
    mano_layer = ManoLayer(mano_root='mano/models', use_pca=False, ncomps=6, flat_hand_mean=True, side=handSelection)
    
    # Convert inputs to tensors and reshape them to be compatible with Mano Layer
    fullpose_tensor = torch.as_tensor(fullpose, dtype=torch.float32).reshape(1, -1)
    shape_tensor = torch.as_tensor(beta, dtype=torch.float32).reshape(1, -1)
    trans_tensor = torch.as_tensor(trans, dtype=torch.float32).reshape(1, -1)

    # Pass to Mano layer
    hand_verts, _ = mano_layer(fullpose_tensor, shape_tensor, trans_tensor)
    
    # return outputs as numpy arrays and scale them
    hand_verts = hand_verts.cpu().detach().numpy()[0] / 1000
    return hand_verts




def load_annotations(fileName=''):

    #hand Pose Mano
    handPose_filename = fileName
    dict = {}
    with open(handPose_filename, "r") as file1:
        f_list = [float(i) for line in file1 for i in line.split(' ') if i.strip()]
        arr = np.array(f_list)

        dict['handTrans_right'] = arr[63:66]
        dict['handPose_right'] = arr[66:114]
        dict['handBeta_right'] = arr[114:124]

        dict['handTrans_left'] = arr[1:4]
        dict['handPose_left'] = arr[4:52]
        dict['handBeta_left'] = arr[52:62]

    return dict

class Dataset(data.Dataset):

    def __init__(self, root='./', load_set='train', transform=None, depth=1):
        self.root = root#os.path.expanduser(root)
        self.transform = transform
        self.load_set = load_set  # 'train','val','test'
        self.depth = depth

        self.images = np.load(os.path.join(root, 'rgbImages_%s.npy'%self.load_set))
        self.depths = np.load(os.path.join(root, 'depthImages_%s.npy'%self.load_set))
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy'%self.load_set))
        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy'%self.load_set))
        self.objMesh = np.load(os.path.join(root, 'objMesh-%s.npy'%self.load_set))

        
        #if shuffle:
        #    random.shuffle(data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """
        path = self.images[index]


        sub_path = path.split("raw/")[1]
        sub_path = sub_path[:-4]+'.txt'
        sub_path = sub_path.replace("rgb","hand_pose_mano")
        mano_path = os.path.join('/netscratch', 'satti', 'mano_labels_v1_1', sub_path)
        anno = load_annotations(mano_path)

        mesh3D_right = load_mesh_from_manolayer(anno['handPose_right'], anno['handBeta_right'], anno['handTrans_right'], 'right')
        mesh3D_left = load_mesh_from_manolayer(anno['handPose_left'], anno['handBeta_left'], anno['handTrans_left'], 'left')


        sub_path = os.path.join('/', self.images[index].split("../")[-1])
        image = cv2.cvtColor(cv2.imread(sub_path), cv2.COLOR_BGR2RGB)

        #<--------------------------check-------------------->
        point2d = self.points2d[index]
        point3d = self.points3d[index]
        objMesh = self.objMesh[index]

        #<----------------------TODO------------------------------------->
        center2d, center3d = calculate_center(point2d, point3d)

        #<---------------------ASK-------------------------------------->
        point3d = point3d - center3d

        point3d = point3d * 1000

        if self.transform is not None:
            image = self.transform(image)


        tsdf = np.array([])
        if self.depth > 0:
            #print("ooooooooooooooooooo")
            depth_path = sub_path.replace('rgb', 'depth')    
            tsdf = run_tsdf(depth_path, center3d, tsdf_channels=self.depth)
            point3d /= (cubicSz / 2)
        

        handMesh = np.concatenate([mesh3D_left, mesh3D_right])
        hand_obj_Mesh = np.concatenate([handMesh, objMesh])
        hand_obj_Mesh = hand_obj_Mesh * 1000

        #print("-------------------------")
        #print(hand_obj_Mesh)
        #print("=========================")
        #print(point3d)

        #visualizeTSDF(tsdf, cubicSz, color=False)
      

        return self.images[index], image[:3], tsdf, point2d, point3d, handMesh, objMesh, hand_obj_Mesh

    def __len__(self):
        return len(self.images)
