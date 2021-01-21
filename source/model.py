import torch.nn.functional as F
import torch.nn as nn


class BinaryClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here
        # defining 2 linear layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(0.3)
        # sigmoid layer
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
       
        out = F.relu(self.fc1(x)) # activation on hidden layer
        out = self.drop(out)
        out = self.fc2(out)
        return self.sig(out) # returning class score
        
        return x