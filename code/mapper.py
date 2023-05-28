import torch

# mapper = mapping network

class AffineMapper(torch.nn.Module) :
    def __init__(self, input_dim, output_dim) :
        super().__init__()
        self.model = torch.nn.Linear(input_dim, output_dim)
    def abandon_grad(self) :
        self.model.requires_grad_ = False
    def forward(self, x) :
        return self.model(x)
    def save(self, path) :
        torch.save(self.state_dict(), path)
    def load(self, path) :
        if path != "None" :
            self.load_state_dict(torch.load(path, map_location = torch.device("cpu")))
        if torch.cuda.is_available() :
            self.model = self.model.cuda()
        self.eval()
    def trained_parameters(self) :
        return self.parameters()