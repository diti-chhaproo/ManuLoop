import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

# re-import dataset loader since this is a separate file
from dataset import load_graphs

train_path = r'C:\manuloop\MFCAD_dataset\MFCAD++_dataset\hierarchical_graphs\training_MFCAD++.h5'
val_path = r'C:\manuloop\MFCAD_dataset\MFCAD++_dataset\hierarchical_graphs\val_MFCAD++.h5'

train_data_list = load_graphs(train_path)
val_data_list = load_graphs(val_path)

train_loader = DataLoader(train_data_list, batch_size=16, shuffle=True, follow_batch=['mesh_x', 'x'])
val_loader = DataLoader(val_data_list, batch_size=16, shuffle=False, follow_batch=['mesh_x', 'x'])

# ---- class weights to handle stock class imbalance ----
all_labels = torch.cat([d.y for d in train_data_list])
class_counts = torch.bincount(all_labels, minlength=25)
class_weights = 1.0 / (class_counts.float() + 1e-6)
class_weights = class_weights / class_weights.sum() * 25

print(f"\n{'='*52}")
print(f"  ManuLoop — Hierarchical GCN Training")
print(f"{'='*52}")
print(f"  Train graphs : {len(train_data_list)}")
print(f"  Val graphs   : {len(val_data_list)}")
print(f"  Device       : {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"{'='*52}")
print(f"\n  Class distribution (0-24):")
for i, count in enumerate(class_counts):
    bar = '█' * int(count / class_counts.max() * 20)
    print(f"  Class {i:02d} | {bar:<20} | {count:>7,}")
print()

# ---- hierarchical GCN ----
class HierarchicalGCN(nn.Module):
    def __init__(self, brep_in=5, mesh_in=4, hidden=128, num_classes=25):
        super().__init__()

        # mesh level: process fine geometry
        self.mesh_conv1 = GCNConv(mesh_in, hidden)
        self.mesh_conv2 = GCNConv(hidden, hidden)
        self.mesh_bn1 = nn.BatchNorm1d(hidden)
        self.mesh_bn2 = nn.BatchNorm1d(hidden)

        # brep level: process face graph with aggregated mesh features
        self.brep_conv1 = GCNConv(brep_in + hidden, hidden)
        self.brep_conv2 = GCNConv(hidden, hidden)
        self.brep_conv3 = GCNConv(hidden, hidden)
        self.brep_bn1 = nn.BatchNorm1d(hidden)
        self.brep_bn2 = nn.BatchNorm1d(hidden)
        self.brep_bn3 = nn.BatchNorm1d(hidden)

        # classification head
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, mesh_x, mesh_edge_index, incidence):
        # 1. process mesh graph
        m = F.relu(self.mesh_bn1(self.mesh_conv1(mesh_x, mesh_edge_index)))
        m = F.dropout(m, p=0.3, training=self.training)
        m = F.relu(self.mesh_bn2(self.mesh_conv2(m, mesh_edge_index)))

        # 2. aggregate mesh embeddings up to brep face level
        # incidence[:, 0] = mesh node indices, incidence[:, 1] = face node indices
        face_count = x.shape[0]
        aggregated = scatter_mean(m[incidence[:, 0]], incidence[:, 1], dim=0, dim_size=face_count)

        # 3. concatenate brep features with aggregated mesh features
        x = torch.cat([x, aggregated], dim=1)

        # 4. process brep face graph
        x = F.relu(self.brep_bn1(self.brep_conv1(x, edge_index)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.brep_bn2(self.brep_conv2(x, edge_index)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.brep_bn3(self.brep_conv3(x, edge_index)))

        # raw logits, one row per face/node
        return self.classifier(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HierarchicalGCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# ---- training loop ----
def train_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(
            batch.x,
            batch.edge_index,
            batch.mesh_x,
            batch.mesh_edge_index,
            torch.stack([batch.incidence_mesh, batch.incidence_face], dim=1)
        )
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_data_list)

# ---- validation accuracy ----
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(
                batch.x,
                batch.edge_index,
                batch.mesh_x,
                batch.mesh_edge_index,
                torch.stack([batch.incidence_mesh, batch.incidence_face], dim=1)
            )
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total

print(f"  {'Epoch':<8} {'Loss':<12} {'Val Acc':<12} {'R-01 Status'}")
print(f"  {'-'*48}")
for epoch in range(1, 51):
    loss = train_epoch()
    val_acc = evaluate(val_loader)
    status = '✓ Target met' if val_acc >= 0.85 else ''
    print(f"  {epoch:<8} {loss:<12.4f} {val_acc:<12.4f} {status}")