import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Client:
    def __init__(self, client_id, images, labels, device='cuda:0', val_ratio=0.2, noise_level=None):
        self.id = client_id
        self.device = device

        # Normalize to [0,1] and reshape (N,1,28,28)
        images = np.array(images, dtype=np.float32) / 255.0
        labels = np.array(labels, dtype=np.int64)
        self.num_samples = len(labels)

        # Shuffle data before splitting train/val
        perm = np.random.permutation(len(labels))
        images = images[perm]
        labels = labels[perm]

        # Add Laplace noise if noise_level is specified
        noise_level = float(noise_level) if noise_level else 0.0
        if noise_level > 0.0:
            noise = np.random.laplace(loc=0.0, scale=noise_level, size=images.shape).astype(np.float32)
            images = images + noise
            # Clip to [0,1] to keep pixel values valid
            images = np.clip(images, 0.0, 1.0)

        split_idx = int(len(labels)*(1-val_ratio))
        self.train_x = torch.from_numpy(images[:split_idx]).unsqueeze(1).to(self.device)
        self.train_y = torch.from_numpy(labels[:split_idx]).long().to(self.device)
        self.val_x   = torch.from_numpy(images[split_idx:]).unsqueeze(1).to(self.device)
        self.val_y   = torch.from_numpy(labels[split_idx:]).long().to(self.device)

    def get_train_loader(self, batch_size=32):
        ds = TensorDataset(self.train_x, self.train_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def get_val_loader(self, batch_size=32):
        ds = TensorDataset(self.val_x, self.val_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def local_train(self, global_state_dict, model_fn, epochs=1, batch_size=32, lr=0.01):
        model = model_fn().to(self.device)
        model.load_state_dict(global_state_dict)
        model.train()

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        loader = self.get_train_loader(batch_size=batch_size)
        for _ in range(epochs):
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()

        # Return updated model state and number of training samples
        updated_state = {k: v.clone() for k, v in model.state_dict().items()} 
        return updated_state, len(self.train_y)
