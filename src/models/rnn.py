import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNNModel(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, layer_dim:int, output_dim:int, dropout_prob:float):
        """Initiate RNN instance.

        Parameters
        ----------
        input_dim : int
            The number of nodes in the input layer
        hidden_dim : int
            The number of nodes in each layer
        layer_dim : int
            The number of layers in the network
        output_dim : int
            The number of nodes in output layer
        dropout_prob : The probability of nodes being dropped out
            The probability of nodes being dropped out
        """
        super(RNNModel, self).__init__()
        
        # # defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        
        # # fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """The forward method takes input tensor x and does forward propagation

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of the shape (batch_size, sequence length, input_dim)
            
        Returns
        -------
        out : torch.Tensor
            The output tensor of the shape (batch size, output_dim)
        """
        # # initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        
        # # forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())
        
        # # reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # # so that it can fit into the fully connected layer
        out = out[:, -1, :]
        
        # # convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        
        return out