from torch import nn


class VisionTransformerMLP(nn.Module):
    def __init__(self,embed_dims, hidden_units):
        super(VisionTransformerMLP, self).__init__()
        self.ed = embed_dims
        self.hu = hidden_units
        self.mlp1 = nn.Linear(self.ed,self.hu)
        self.mlp2 = nn.Linear(self.hu,self.ed)
        self.actv = nn.GELU()
