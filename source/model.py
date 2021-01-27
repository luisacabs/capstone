import torch.nn.functional as F
import torch.nn as nn


class BinaryClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super(BinaryClassifier, self).__init__()

        # defining 2 linear layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # dropout layer
        self.drop = nn.Dropout(0.3)
        # sigmoid layer
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
       
        # activation function on hidden layer
        out = F.relu(self.fc1(x)) 
        out = self.drop(out)
        out = self.fc2(out)
        # returning class score
        return self.sig(out) 
        