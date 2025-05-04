import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import numpy as np
import os
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import pandas as pd
from datetime import datetime

current_dir = os.path.dirname(__file__)

element_map = {'H': 0.0, 'HE': 0.00847457627118644, 'LI': 0.01694915254237288, 'BE': 0.025423728813559324, 'B': 0.03389830508474576, 'C': 0.0423728813559322, 'N': 0.05084745762711865, 'O': 0.059322033898305086, 'F': 0.06779661016949153, 'NE': 0.07627118644067797, 'NA': 0.0847457627118644, 'MG': 0.09322033898305085, 'AL': 0.1016949152542373, 'SI': 0.11016949152542373, 'P': 0.11864406779661017, 'S': 0.1271186440677966, 'CL': 0.13559322033898305, 'AR': 0.1440677966101695, 'K': 0.15254237288135594, 'CA': 0.16101694915254236, 'SC': 0.1694915254237288, 'TI': 0.17796610169491525, 'V': 0.1864406779661017, 'CR': 0.19491525423728814, 'MN': 0.2033898305084746, 'FE': 0.211864406779661, 'CO': 0.22033898305084745, 'NI': 0.2288135593220339, 'CU': 0.23728813559322035, 'ZN': 0.2457627118644068, 'GA': 0.2542372881355932, 'GE': 0.2627118644067797, 'AS': 0.2711864406779661, 'SE': 0.2796610169491525, 'BR': 0.288135593220339, 'KR': 0.2966101694915254, 'RB': 0.3050847457627119, 'SR': 0.3135593220338983, 'Y': 0.3220338983050847, 'ZR': 0.3305084745762712, 'NB': 0.3389830508474576, 'MO': 0.3474576271186441, 'TC': 0.3559322033898305, 'RU': 0.3644067796610169, 'RH': 0.3728813559322034, 'PD': 0.3813559322033898, 'AG': 0.3898305084745763, 'CD': 0.3983050847457627, 'IN': 0.4067796610169492, 'SN': 0.4152542372881356, 'SB': 0.423728813559322, 'TE': 0.4322033898305085, 'I': 0.4406779661016949, 'XE': 0.4491525423728814, 'CS': 0.4576271186440678, 'BA': 0.4661016949152542, 'LA': 0.4745762711864407, 'CE': 0.4830508474576271, 'PR': 0.4915254237288136, 'ND': 0.5, 'PM': 0.5084745762711864, 'SM': 0.5169491525423728, 'EU': 0.5254237288135594, 'GD': 0.5338983050847458, 'TB': 0.5423728813559322, 'DY': 0.5508474576271186, 'HO': 0.559322033898305, 'ER': 0.5677966101694916, 'TM': 0.576271186440678, 'YB': 0.5847457627118644, 'LU': 0.5932203389830508, 'HF': 0.6016949152542372, 'TA': 0.6101694915254238, 'W': 0.6186440677966102, 'RE': 0.6271186440677966, 'OS': 0.635593220338983, 'IR': 0.6440677966101694, 'PT': 0.652542372881356, 'AU': 0.6610169491525424, 'HG': 0.6694915254237288, 'TL': 0.6779661016949152, 'PB': 0.6864406779661016, 'BI': 0.6949152542372882, 'PO': 0.7033898305084746, 'AT': 0.711864406779661, 'RN': 0.7203389830508474, 'FR': 0.7288135593220338, 'RA': 0.7372881355932204, 'AC': 0.7457627118644068, 'TH': 0.7542372881355932, 'PA': 0.7627118644067796, 'U': 0.7711864406779662, 'NP': 0.7796610169491526, 'PU': 0.788135593220339, 'AM': 0.7966101694915254, 'CM': 0.8050847457627118, 'BK': 0.8135593220338984, 'CF': 0.8220338983050848, 'ES': 0.8305084745762712, 'FM': 0.8389830508474576, 'MD': 0.847457627118644, 'NO': 0.8559322033898306, 'LR': 0.864406779661017, 'RF': 0.8728813559322034, 'DB': 0.8813559322033898, 'SG': 0.8898305084745762, 'BH': 0.8983050847457628, 'HS': 0.9067796610169492, 'MT': 0.9152542372881356, 'DS': 0.923728813559322, 'RG': 0.9322033898305084, 'CN': 0.940677966101695, 'NH': 0.9491525423728814, 'FL': 0.9576271186440678, 'MC': 0.9661016949152542, 'LV': 0.9745762711864406, 'TS': 0.9830508474576272, 'OG': 0.9915254237288136, 'DU': 1.0}

def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val == min_val:
        return np.ones_like(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def normalize_points(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    rotation_matrix = pca.components_
    return np.dot(centered_points, rotation_matrix.T)

def discover(hull_points_indices, points):
    i, j = np.triu_indices(len(hull_points_indices), k=1)
    distances = np.linalg.norm(points[hull_points_indices[i]] - points[hull_points_indices[j]], axis=1)
    max_idx = np.argmax(distances)
    index1 = hull_points_indices[i[max_idx]]
    index2 = hull_points_indices[j[max_idx]]
    return index1, index2

def rotationMatrix(a, theta):
    a = a / np.linalg.norm(a)
    C = np.cos(theta)
    S = np.sin(theta)
    one_minus_C = 1.0 - C
    n_x, n_y, n_z = a
    R = np.array([
        [C + n_x**2 * one_minus_C,      n_x * n_y * one_minus_C - n_z * S,  n_x * n_z * one_minus_C + n_y * S],
        [n_y * n_x * one_minus_C + n_z * S,  C + n_y**2 * one_minus_C,      n_y * n_z * one_minus_C - n_x * S],
        [n_z * n_x * one_minus_C - n_y * S,  n_z * n_y * one_minus_C + n_x * S,  C + n_z**2 * one_minus_C]
    ])
    return R

def rotation(points,point1,point2):
    A = point1
    points -= A
    B = point2
    Y = np.array( [0.0, 1.0, 0.0 ] )
    axis = np.cross( B, Y )
    mag = np.linalg.norm( axis )
    if ( abs(mag) < 1.0e-20 ): axis = np.array([1.0,0,0])
    sintheta = mag / np.linalg.norm( B )
    theta = np.arcsin(sintheta)
    if np.dot( B, Y ) < 0: theta = np.pi - theta
    R = rotationMatrix( axis, theta )
    points = ( R @ (points.T) ).T
    return points

def CheckP(points):
    if len(points) < 3:
        return True
    p0 = points[0]
    diff_vectors = points - p0
    rank = np.linalg.matrix_rank(diff_vectors)
    return rank < 3
    
def C(points):
    if CheckP(points):
        return None, None, None
    else:
        hull = ConvexHull(points)
    
    hull_points_indices = hull.vertices
    index1, index2 = discover(hull_points_indices, points)
    A = np.array(points[index1])
    B = np.array(points[index2])
    mask = np.ones(len(points), dtype=bool)
    mask[index1] = False
    mask[index2] = False
    PA = points[mask] - A
    BA = B - A
    cross_products = np.cross(PA, BA)
    norm_cross_products = np.linalg.norm(cross_products, axis=1)
    norm_BA = np.linalg.norm(BA)
    distances = norm_cross_products / norm_BA
    distances = np.insert(distances, index1, 0)
    distances = np.insert(distances, index2, 0)
    dot_products = np.dot(PA, BA)
    norm_PA = np.linalg.norm(PA, axis=1)
    cos_theta = dot_products / (norm_PA * norm_BA)
    angles_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angles = np.degrees(angles_radians)
    angles = np.insert(angles, index1, 0)
    angles = np.insert(angles, index2, 0)
    points = normalize_points(points)
    points = rotation(points,points[index1],points[index2])
    points = normalize_array(points)
    distances = normalize_array(distances)
    angles = normalize_array(angles)
    return points, distances, angles
    
def DRUG(SMILES): 
    m = Chem.MolFromSmiles(SMILES)
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xf00d
    params.useSmallRingTorsions = True
    params.useExpTorsionAnglePrefs = True
    status = AllChem.EmbedMolecule(m, params)
    if status == -1:
        params.useRandomCoords = True
        params.maxAttempts = 1000
        status = AllChem.EmbedMolecule(m, params)

    try:
        conf = m.GetConformer()
    except:
        return None
    
    bond_types = {
        Chem.rdchem.BondType.SINGLE: 0.25,
        Chem.rdchem.BondType.DOUBLE: 0.5,
        Chem.rdchem.BondType.TRIPLE: 0.75,
        Chem.rdchem.BondType.AROMATIC: 1.0
    }
    
    edge_attr = []
    edge_index = []
    if len(m.GetBonds()) == 0:
        return None
    for bond in m.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        edge_index.append([begin_idx, end_idx])
        edge_index.append([end_idx, begin_idx])
        
        bond_type = bond.GetBondType()
        is_in_ring = bond.IsInRing()
        bond_order = bond.GetBondTypeAsDouble() / 3.0
        edge_attr.append([bond_types[bond_type], is_in_ring, bond_order])
        edge_attr.append([bond_types[bond_type], is_in_ring, bond_order])

    edge_index = np.array(edge_index)
    
    Position = []
    atom_names = []
    for i, atom in enumerate(m.GetAtoms()):
        atom_symbol = atom.GetSymbol()
        pos = conf.GetAtomPosition(i)
        Pos = pos.x, pos.y, pos.z
        Position.append(Pos)
        atom_names.append(element_map[atom_symbol.upper()])
    
    Position = np.array(Position, float)
    # num_nodes = Position.shape[0]
    # adj_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    # for edge in edge_index:
    #     i, j = edge
    #     adj_matrix[i, j] = 1
    #     adj_matrix[j, i] = 1
    # edge_index = torch.tensor(adj_matrix)
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    
    Position, Distances, Angles = C(Position)
    if Position is not None:
        atom_names = np.array(atom_names).reshape(-1, 1)
        Distances = np.array(Distances).reshape(-1, 1)
        Angles = np.array(Angles).reshape(-1, 1)
        
        atom_features = np.hstack((Position, Distances, Angles, atom_names))
        
        X = torch.tensor(atom_features, dtype=torch.float, requires_grad=True)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        if edge_attr.max() > 1:
            print(edge_attr.max())
            quit()
        graph_data = Data(x=X, edge_index=edge_index, edge_attr= edge_attr)
        return graph_data
    else:
        return None

import csv 
def start():
    Dtest = pd.read_csv("test.csv")
    Dtrain = pd.read_csv("train.csv")
    Dval = pd.read_csv("val.csv")
    DAVIS = pd.concat([Dtest, Dtrain, Dval], ignore_index=True)
    DF = DAVIS["SMILES"].drop_duplicates()

    listo = []
    for i in range(68):
        SMILES = DF.iloc[i]
    #     listo.append([f"sequence_{i}", SMILES])
    # file_name = "D.csv"
    # with open(file_name, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['ID', 'Sequence'])
    #     writer.writerows(listo)
        
        DATA = DRUG(SMILES)
        if DATA is not None:
            print(i, DATA)
            torch.save(DATA, f'D/D{i}.pt')
        else:
            print(i)
    
if __name__ == "__main__":
    start()