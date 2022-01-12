# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
from PIL import Image
import torch
from manopth.manolayer import ManoLayer


"""# Load Dataset"""

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

    def __init__(self, root='./', load_set='train', transform=None):
        self.root = root#os.path.expanduser(root)
        self.transform = transform
        self.load_set = load_set  # 'train','val','test'

        #self.root = 'dataset/ho'
        #print('roottt', root)
        #print('aaa---------------',os.path.join(self.root, 'images-%s.npy'%self.load_set))

        self.images = np.load(os.path.join(root, 'rgbImages_%s.npy'%self.load_set))
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy'%self.load_set))
        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy'%self.load_set))
        #print(self.images)
        
        #if shuffle:
        #    random.shuffle(data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """
        #print("============================",index)
        path = self.images[index]
        sub_path = path.split("raw/")[1]
        sub_path = sub_path[:-4]+'.txt'
        sub_path = sub_path.replace("rgb","hand_pose_mano")
        #print("--->",sub_path)
        mano_path = os.path.join('/home', 'satti', 'mano_labels_v1_1', sub_path)
        anno = load_annotations(mano_path)

        mesh3D_right = load_mesh_from_manolayer(anno['handPose_right'], anno['handBeta_right'], anno['handTrans_right'], 'right')
        mesh3D_left = load_mesh_from_manolayer(anno['handPose_left'], anno['handBeta_left'], anno['handTrans_left'], 'left')

        mesh = np.concatenate([mesh3D_left, mesh3D_right])
        #print("oooooooo",index)
        #print('.......opening an image: ', self.images[index])
        sub_path = self.images[index].split("../")[-1]
        image = Image.open(os.path.join('/', sub_path))
        point2d = self.points2d[index]
        point3d = self.points3d[index]

        if self.transform is not None:
            image = self.transform(image)

        return image[:3], point2d, point3d, mesh

    def __len__(self):
        return len(self.images)
