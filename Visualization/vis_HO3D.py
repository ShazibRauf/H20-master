"""
Visualize the projections in published HO-3D dataset
"""
from os.path import join
import pip
import argparse
from utils.vis_utils import *
import random
from copy import deepcopy
import open3d
from manopth.manolayer import ManoLayer
import torch
import sys

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        from pip._internal.main import main as pipmain
        pipmain(['install', package])

try:
    import matplotlib.pyplot as plt
except:
    install('matplotlib')
    import matplotlib.pyplot as plt

try:
    import chumpy as ch
except:
    install('chumpy')
    import chumpy as ch


try:
    import pickle
except:
    install('pickle')
    import pickle

import cv2
from mpl_toolkits.mplot3d import Axes3D

MANO_MODEL_PATH = 'mano/models'

# mapping of joints from MANO model order to simple order(thumb to pinky finger)
jointsMapManoToSimple = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
#jointsMapManoToSimple = np.array([0, 5,6,7,9,10,11,17,18,19,13,14,15,1,2,3,4,8,12,16,20])


#if not os.path.exists(MANO_MODEL_PATH):
#    raise Exception('MANO model missing! Please run setup_mano.py to setup mano folder')
#else:
#    from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model


def forwardKinematics(fullpose, trans, beta, mano_mode, handSelection):
    '''
    MANO parameters --> 3D pts, mesh
    :param fullpose:
    :param trans:
    :param beta:
    :return: 3D pts of size (21,3)
    '''

    assert fullpose.shape == (48,)
    assert trans.shape == (3,)
    assert beta.shape == (10,)

    if mano_mode == 'model':
        m = load_model(join(MANO_MODEL_PATH,'MANO_'+handSelection.upper()+'.pkl'), ncomps=6, flat_hand_mean=True)
        m.fullpose[:] = fullpose
        m.trans[:] = trans
        m.betas[:] = beta
        return m.J_transformed.r, m, None
    
    else:
        mano_layer = ManoLayer(mano_root='mano/models', use_pca=False, ncomps=6, flat_hand_mean=True, side = handSelection)

        # Convert inputs to tensors and reshape them to be compatible with Mano Layer
        fullpose_tensor = torch.as_tensor(fullpose, dtype=torch.float32).reshape(1, -1)
        shape_tensor = torch.as_tensor(beta, dtype=torch.float32).reshape(1, -1)
        trans_tensor = torch.as_tensor(trans, dtype=torch.float32).reshape(1, -1)

        # Pass to Mano layer
        hand_verts, hand_joints = mano_layer(fullpose_tensor, shape_tensor, trans_tensor)

        # return outputs as numpy arrays
        hand_verts = hand_verts.cpu().detach().numpy()[0]
        hand_joints = hand_joints.cpu().detach().numpy()[0]

        return hand_joints, hand_verts, mano_layer.th_faces


if __name__ == '__main__':

    # parse the input arguments
    baseDir = sys.argv[1]
    split = 'train'
    args_seq = ""
    args_id = ""
    args_visType = "matplotlib" 

    # some checks to decide if visualizing one single image or randomly picked images
    if args_seq is "":
        args_seq = sys.argv[2]
        print("]]]]]]]]]]]]]]", args_seq)
        runLoop = True
    else:
        runLoop = False

    if args_id is "":
        args_id = sys.argv[3]
    else:
        pass

    if args_visType == "matplotlib":
        o3dWin = Open3DWin()

    while(True):
        seqName = args_seq
        id = args_id

        # read image, depths maps and annotations
        img = read_RGB_img(baseDir, seqName, id, split)
        depth = read_depth_img(baseDir, seqName, id, split)
        anno = read_annotation(baseDir, seqName, id, split)

        if anno['camMatrix'] is None:
            print('Frame %s in sequence %s does not have annotations'%(args_id, args_seq))
            if not runLoop:
                break
            else:
                args_seq = random.choice(os.listdir(join(baseDir, split)))
                args_id = random.choice(os.listdir(join(baseDir, split, args_seq, 'rgb'))).split('.')[0]
                continue

        # get object 3D corner locations for the current pose
        objCorners = anno['objCorners3DRest']
        objCornersTrans = objCorners

        pred_objCorners = anno['pred_objCorners3DRest']
        pred_objCornersTrans = pred_objCorners


        # get the hand Mesh from MANO model for the current pose
        if split == 'train':
            handJoints3D_right = anno['handPose_right']
            handJoints3D_left = anno['handPose_left']
            #print(handJoints3D_left)

            order_idx = np.argsort(jointsMapManoToSimple)
            #handJoints3D_left = handJoints3D_left[order_idx]
            #handJoints3D_right = handJoints3D_right[order_idx]


            pred_handJoints3D_right = anno['pred_handPose_right']
            pred_handJoints3D_left = anno['pred_handPose_left']

            #pred_handJoints3D_right = pred_handJoints3D_right[order_idx]
            #pred_handJoints3D_left = pred_handJoints3D_left[order_idx]


        # project to 2D
        if split == 'train':
            handKps_right = project_3D_points(anno['camMat'], handJoints3D_right, is_OpenGL_coords=True)
            handKps_left = project_3D_points(anno['camMat'], handJoints3D_left, is_OpenGL_coords=True)

            pred_handKps_right = project_3D_points(anno['camMat'], pred_handJoints3D_right, is_OpenGL_coords=True)
            pred_handKps_left = project_3D_points(anno['camMat'], pred_handJoints3D_left, is_OpenGL_coords=True)


        else:
            # Only root joint available in evaluation split
            handKps = project_3D_points(anno['camMat'], np.expand_dims(anno['handJoints3D'],0), is_OpenGL_coords=True)

        objKps = project_3D_points(anno['camMat'], objCornersTrans, is_OpenGL_coords=True)
        pred_objKps = project_3D_points(anno['camMat'], pred_objCornersTrans, is_OpenGL_coords=True)




        if args_visType == 'matplotlib':
            # draw 2D projections of annotations on RGB image
            
            imgAnno = showHandJoints(img, handKps_right)
            imgAnno = showHandJoints(imgAnno, handKps_left)
            imgAnno = showObjJoints(imgAnno, objKps, lineThickness=2)


            pred_imgAnno = showHandJoints(img, pred_handKps_right)
            pred_imgAnno = showHandJoints(pred_imgAnno, pred_handKps_left)
            pred_imgAnno = showObjJoints(pred_imgAnno, pred_objKps, lineThickness=2)


            # create matplotlib window
            fig = plt.figure(figsize=(2, 3))
            figManager = plt.get_current_fig_manager()
            figManager.resize(1000, 1000)

            # show RGB image
            ax0 = fig.add_subplot(2, 3, 1)
            ax0.imshow(img[:, :, [2, 1, 0]])
            ax0.title.set_text('RGB Image')


            # show 2D projections of annotations on RGB image
            ax1 = fig.add_subplot(2, 3, 2)
            ax1.imshow(imgAnno[:, :, [2, 1, 0]])
            ax1.title.set_text('Orignal (3D ---> 2D)')


            # show 2D projections of annotations on RGB image
            ax2 = fig.add_subplot(2, 3, 3)
            ax2.imshow(pred_imgAnno[:, :, [2, 1, 0]])
            ax2.title.set_text('Predicted (3D ---> 2D)')


            # show depth map
            ax3 = fig.add_subplot(2, 3, 4)
            im = ax3.imshow(depth)
            ax3.title.set_text('Depth Map')


            ax4 = fig.add_subplot(2, 3, 5, projection="3d")
            show3DHandJoints(ax4, handJoints3D_left)
            show3DHandJoints(ax4, handJoints3D_right)
            show3DObjCorners(ax4, anno['objCorners3DRest'])
            ax4.title.set_text('Orignal 3D')


            ax5 = fig.add_subplot(2, 3, 6, projection="3d")
            show3DHandJoints(ax5, pred_handJoints3D_left)
            show3DHandJoints(ax5, pred_handJoints3D_right)
            show3DObjCorners(ax5, anno['pred_objCorners3DRest'])
            ax5.title.set_text('Predicted 3D')


            plt.show()       

        else:
            raise Exception('Unknown visualization type')

        if runLoop:
            args_seq = random.choice(os.listdir(join(baseDir, split)))
            args_id = random.choice(os.listdir(join(baseDir, split, args_seq, 'rgb'))).split('.')[0]
        else:
            break
