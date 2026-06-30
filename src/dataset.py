import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import h5py
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_graphs(h5_path):
    data_list = []
    with h5py.File(h5_path, 'r') as f:
        for key in list(f.keys()):
            batch = f[key]

            #there are 5 numbers per face, the surface area, centroid x and y and z, surface type
            node_features = torch.tensor(batch['V_1'][:], dtype=torch.float)

            #pulls which of the 25 feature classes the face belongs to
            labels = torch.tensor(batch['labels'][:], dtype=torch.long)

            #pulls up which faces are connected to which
            edge_index = torch.tensor(batch['A_1_idx'][:], dtype=torch.long).t().contiguous()

            #packages into one PyG Data object
            data = Data(x=node_features, edge_index=edge_index, y=labels)
            data_list.append(data)
    return data_list

train_path = r'C:\manuloop\MFCAD_dataset\MFCAD++_dataset\hierarchical_graphs\training_MFCAD++.h5'
val_path = r'C:\manuloop\MFCAD_dataset\MFCAD++_dataset\hierarchical_graphs\val_MFCAD++.h5'

train_data_list = load_graphs(train_path)
val_data_list = load_graphs(val_path)

print(f"Train graphs: {len(train_data_list)}")
print(f"Val graphs: {len(val_data_list)}")
print(f"Example train graph: {train_data_list[0]}")

train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data_list, batch_size=32, shuffle=False)

for batch in train_loader:
    print(f"One training batch: {batch}")
    break