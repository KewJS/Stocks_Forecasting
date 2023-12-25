import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GRUModel(nn.Module):
    """GRUModel class extends nn.Module class and works as a constuctor for GRUs.
    
    GRUModel class initiates a GRU module based on Pytorch's nn.Module class. It has
    only two methods, namely init() and forward(). While the init() method intiates
    the model with the given input parameters, the forward() method defines how the 
    forward propagation need to be calculated. Since Pytorch automatically defined
    back propagation, there is no need to define back propagation method.

    Attributes
    ----------
    hidden_dim : int
        The number of nodes in each layer
    layer_dim : int
        The number of layers in the network
    lstm : nn.LSTM
        The LSTM model constructed with the input parameters.
    fc : nn.Linear
        The fully connected layer to convert the final state of LSTMs to our desired output shape.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        """Initiate GRU instance.

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
        super(GRUModel, self).__init__()
        
        # # defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        
        # # GRU layer
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        
        # # fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """The forward method takes input tensor x and does forward propagation

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of the shape (batch size, sequence length, input_dim)

        Returns
        -------
        torch.Tensor
            The output tensor of the shape (batch size, output_dim)
        """
        # # initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        
        # # forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())
        
        # # reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # # so that it can fit into the fully connected layer
        out = out[:, -1, :]
        
        # # convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        
        return out