import numpy as np
import random
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from server import Server
from client import Client
from clientActor import clientActor 
from model import model_fn

import ray


def main(num_rounds, c, local_epochs, batch_size, lr, device='cuda:0', parallel=False, noise_level=0.0):

    # Load data
    train_data = np.load('data/train_data.npy', allow_pickle=True)
    test_data = np.load('data/test_data.npy', allow_pickle=True)

    # Convert test_data[0] to torch DataLoader
    x_test = torch.from_numpy(np.array(test_data[0]['images'], dtype=np.float32)/255.0).unsqueeze(1).to(device)
    y_test = torch.from_numpy(np.array(test_data[0]['labels'], dtype=np.int64)).to(device)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=128, shuffle=False)

    # Create clients
    client_device = 'cpu' if parallel else device
    clients = {}
    for cid in range(len(train_data)):
        clients[cid] = Client(cid, train_data[cid]['images'], train_data[cid]['labels'],
                               device=client_device, noise_level=noise_level)

    # Set up FedAvg training
    if not parallel:
        clients_per_round = max(1, int(c * len(clients)))
    else:
        clients_per_round = 4
        # Ray setup
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    # Create central server
    global_model = model_fn()
    server = Server(global_model, device=device)

    train_acc_list = []
    val_acc_list = []

    for r in range(1, num_rounds+1):
        # Sample clients for this round
        selected = random.sample(list(clients.keys()), clients_per_round)
        if not parallel:
            updates = []
            global_weights = server.get_global_weights()
            # Sequential local training
            for cid in selected:
                updated_state, n_samples = clients[cid].local_train(global_weights, model_fn,
                                                                        epochs=local_epochs,
                                                                        batch_size=batch_size,
                                                                        lr=lr)
                updates.append((updated_state, n_samples))

        else:
            global_weights = server.get_global_weights()
            # Create a Ray actor for each selected client only
            remote_clients = [
                clientActor.remote(cid, train_data[cid]['images'], train_data[cid]['labels'], device)
                for cid in selected
            ]
            # Parallel local training: each `.remote` returns a future
            update_futures = [
                rc.local_train.remote(global_weights, model_fn, local_epochs, batch_size, lr)
                for rc in remote_clients
            ]
            updates = ray.get(update_futures)


        # Aggregate updates
        server.aggregate(updates)

        # Periodic evaluation
        if r % 10 == 0 or r == 1:
            # Evaluate aggregated model on validation (all client validation sets)
            total_correct = 0
            total_samples = 0
            total_loss = 0.0
            for cid in clients:
                val_loader = clients[cid].get_val_loader(batch_size=128)
                _, avg_loss, correct, n = server.evaluate(val_loader)
                total_correct += correct
                total_samples += n
                if avg_loss is not None:
                    total_loss += avg_loss * n  # accumulate sum(loss*count)
            val_acc = total_correct / total_samples if total_samples > 0 else 0.0
            val_acc_list.append(val_acc)

            # Evaluate on training (all client train sets)
            total_correct = 0
            total_samples = 0
            total_loss = 0.0
            for cid in clients:
                train_loader = clients[cid].get_train_loader(batch_size=128)
                acc, avg_loss, correct, n = server.evaluate(train_loader)
                total_correct += correct
                total_samples += n
                if avg_loss is not None:
                    total_loss += avg_loss * n
            train_acc = total_correct / total_samples if total_samples > 0 else 0.0
            train_acc_list.append(train_acc)

        # Print out train & val accuracy every 50 rounds
        if r % 50 == 0 or r == 1:
            print(f"[Round {r}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Evaluate on test set
    test_acc, _, _, _ = server.evaluate(test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Build plot to show training/validation accuracy
    plt.figure(figsize=(8,5))

    # Define x_axis based on logging interval
    logging_interval = 10 
    x_axis = x_axis = [i * logging_interval - (logging_interval - 1) for i in range(1, len(train_acc_list)+1)]

    plt.plot(x_axis, train_acc_list, label='Train Accuracy')
    plt.plot(x_axis, val_acc_list, label='Validation Accuracy')
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("FedAvg Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # Save plot with descriptive filename
    if parallel:
        prefix = f"runs/parallel={parallel}_{num_rounds}rounds_{local_epochs}epochs_{batch_size}bs_{lr}lr"
    elif noise_level != 0.0:
        prefix = f"runs/noise={noise_level}_{num_rounds}rounds_{c}C_{local_epochs}epochs_{batch_size}bs_{lr}lr"
    else: 
        prefix = f"runs/{num_rounds}rounds_{c}C_{local_epochs}epochs_{batch_size}bs_{lr}lr"
    plt.savefig(f"{prefix}.png")
    plt.close()

    return test_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel execution with Ray")
    parser.add_argument("--server_device", type=str, default="cuda:0",
                        help="Device for server aggregation")
    parser.add_argument("--num_rounds", type=int, default=800,
                        help="Number of global FL rounds")
    parser.add_argument("--c", type=float, default=0.1,
                        help="Client sampling percentage (0–1)")
    parser.add_argument("--local_epochs", type=int, default=10,
                        help="Number of client local epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Client batch size")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate")
    parser.add_argument("--noise_level", type=float, default=0.0,
                        help="Noise level for differential privacy")

    args = parser.parse_args()

    main(num_rounds=args.num_rounds,
         c=args.c,
         local_epochs=args.local_epochs,
         batch_size=args.batch_size,
         lr=args.lr, 
         device=args.server_device,
         parallel=args.parallel,
         noise_level=args.noise_level)