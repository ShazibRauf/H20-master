import numpy as np
import scipy.sparse
from manopth.manolayer import ManoLayer
from scipy.spatial import cKDTree
import os
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch

def createAdjMatrix(faces):
    n = np.max(faces) + 1
    matrix = np.zeros((n, n))
    for face in faces:
        n1 = face[0]
        n2 = face[1]
        n3 = face[2]
        matrix[n1][n2] = 1
        matrix[n2][n1] = 1

        matrix[n1][n3] = 1
        matrix[n3][n1] = 1

        matrix[n2][n3] = 1
        matrix[n3][n2] = 1

    # Adding self connections 
    I = np.identity(n)
    matrix += I

    sparse_matrix = scipy.sparse.csc_matrix(matrix)
    return sparse_matrix
    # scipy.sparse.save_npz(f'datasets/ho/adj_matrix/adj_matrix_{n}.npz', sparse_matrix)


def normalize(point_cloud):
    point_cloud_normalized = (point_cloud - np.min(point_cloud, axis=0)) / (np.max(point_cloud, axis=0) - np.min(point_cloud,axis=0))
    return point_cloud_normalized


def deformMesh(mesh_vertices, point_cloud):
    point_cloud_normalized = normalize(point_cloud)
    mesh_normalized = normalize(mesh_vertices)

    tree = cKDTree(mesh_normalized)

    correspondance = {}
    fail_count = 0

    graph = np.zeros((mesh_vertices.shape[0], point_cloud.shape[0]))
    print(graph.shape)

    # graph construction
    for i, vertex in enumerate(mesh_normalized):
        # print(point_cloud_normalized-vertex)
        dist = np.linalg.norm(point_cloud_normalized - vertex, axis=1)
        # print(dist.shape)
        graph[i] = dist
        # for j, point in enumerate(point_cloud_normalized):
        #     dist = np.linalg.norm(vertex-point)
        #     graph[i][j] = dist

    row_ind, col_ind = linear_sum_assignment(graph)

    for i in range(row_ind.shape[0]):
        mesh_vertices[row_ind[i]] = point_cloud[col_ind[i]]
    return mesh_vertices

""" Load Hand Mesh """

#mano_layer = ManoLayer(mano_root='../mano_v1_2/models', use_pca=False, ncomps=6, flat_hand_mean=True, side = 'left')
#handVerts = mano_layer.th_v_template.cpu().detach().numpy()[0]
#handFaces = mano_layer.th_faces.cpu().detach().numpy()


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#hand Pose Mano
handPose_filename = '/home5/satti/mano_labels_v1_1/subject1/h1/0/cam0/hand_pose_mano/000000.txt'
with open(handPose_filename, "r") as file1:
    f_list = [float(i) for line in file1 for i in line.split(' ') if i.strip()]
    arr = np.array(f_list)

    handTrans_right = arr[63:66]
    handPose_right = arr[66:114]
    handBeta_right = arr[114:124]

    handTrans_left = arr[1:4]
    handPose_left = arr[4:52]
    handBeta_left = arr[52:62]

print("--------------------")
print(handTrans_left)
print(handPose_left)
print(handBeta_left)

#left hand
mano_layer = ManoLayer(mano_root='mano/models', use_pca=False, ncomps=6, flat_hand_mean=True, side = 'left')

# Convert inputs to tensors and reshape them to be compatible with Mano Layer
#fullpose_tensor = torch.as_tensor(handPose_left, dtype=torch.float32).reshape(1, -1)
#shape_tensor = torch.as_tensor(handBeta_left, dtype=torch.float32).reshape(1, -1)
#trans_tensor = torch.as_tensor(handTrans_left, dtype=torch.float32).reshape(1, -1)

#print("00000000000000000000000000000000000000000000000")
#print(fullpose_tensor)
# Pass to Mano layer
#hand_verts, hand_joints = mano_layer(fullpose_tensor, shape_tensor, trans_tensor)

print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")

handVerts = mano_layer.th_v_template.cpu().detach().numpy()[0]
handFaces = mano_layer.th_faces.cpu().detach().numpy()


sparse_matrix = createAdjMatrix(handFaces)
print(sparse_matrix)
scipy.sparse.save_npz(f'handAdjacency_lefthand.npz', sparse_matrix)



#right hand
mano_layer = ManoLayer(mano_root='mano/models', use_pca=False, ncomps=6, flat_hand_mean=True, side = 'right')

# Convert inputs to tensors and reshape them to be compatible with Mano Layer
#fullpose_tensor = torch.as_tensor(handPose_right, dtype=torch.float32).reshape(1, -1)
#shape_tensor = torch.as_tensor(handBeta_right, dtype=torch.float32).reshape(1, -1)
#trans_tensor = torch.as_tensor(handTrans_right, dtype=torch.float32).reshape(1, -1)

# Pass to Mano layer
#hand_verts, hand_joints = mano_layer(fullpose_tensor, shape_tensor, trans_tensor)

handVerts = mano_layer.th_v_template.cpu().detach().numpy()[0]
handFaces = mano_layer.th_faces.cpu().detach().numpy()


sparse_matrix = createAdjMatrix(handFaces)
scipy.sparse.save_npz(f'handAdjacency_righthand.npz', sparse_matrix)

