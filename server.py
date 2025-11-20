import torch

class Server:
    def __init__(self, model, device='cuda:0'):
        self.global_model = model.to(device)
        self.device = device

    def get_global_weights(self):
        return {k: v.clone() for k, v in self.global_model.state_dict().items()}

    def set_global_weights(self, state_dict):
        self.global_model.load_state_dict(state_dict)

    def aggregate(self, client_weights_and_sizes):

        total_samples = sum(n for _, n in client_weights_and_sizes)
        if total_samples == 0:
            return

        agg_state = {}
        for k in client_weights_and_sizes[0][0].keys():
            agg_state[k] = torch.zeros_like(client_weights_and_sizes[0][0][k], device=self.device)

        for state, n in client_weights_and_sizes:
            weight = n / total_samples
            for k in state.keys():
                agg_state[k] += state[k].to(self.device) * weight

        self.set_global_weights(agg_state)

    def evaluate(self, data_loader, criterion=None):
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        device = self.device
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(device).float()
                y = y.to(device).long()
                out = self.global_model(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                if criterion is not None:
                    total_loss += criterion(out, y).item() * y.size(0)
        acc = (correct / total) if total > 0 else 0.0
        avg_loss = (total_loss / total) if total > 0 else None
        return acc, avg_loss, correct, total

    def save(self, path):
        torch.save(self.global_model.state_dict(), path)
