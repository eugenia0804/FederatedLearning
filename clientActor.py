import ray
from client import Client

# Define a Ray remote actor for the client
@ray.remote(num_cpus=1, num_gpus=1)
class clientActor:
    def __init__(self, cid, images, labels, device):
        self.client = Client(cid, images, labels, device=device)

    def local_train(self, global_weights, model_fn, epochs, batch_size, lr):
        return self.client.local_train(global_weights, model_fn, epochs, batch_size, lr)
    
    def get_val_loader(self, batch_size):
        return self.client.get_val_loader(batch_size)
    
    def get_train_loader(self, batch_size):
        return self.client.get_train_loader(batch_size)
    
    def get_val_y_len(self):
        return len(self.client.val_y)
    
    def get_train_y_len(self):
        return len(self.client.train_y)