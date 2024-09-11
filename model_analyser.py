import torch.onnx 

import numpy as np
import torch
import torch.nn as nn
# Define the hyperparameters
input_size = 2
hidden_size = 120
num_stacked_layers = 5
fc_size1 = 140
fc_size2 = 280
fc_size3 = 360
output_size = 2
dropout = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.residual = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out, (hn, cn) = self.lstm(x)
        return out + residual, (hn, cn)

class ResidualFC(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        out = self.fc(x)
        return self.relu(out + residual)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, dropout, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        # Stacked LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList([
            ResidualLSTM(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_stacked_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

        # Separate fully connected layers for each output with residual connections
        self.fc_output1 = nn.Sequential(
            ResidualFC(hidden_size, fc_size1),
            ResidualFC(fc_size1, fc_size2),
            ResidualFC(fc_size2, fc_size3),
            nn.Linear(fc_size3, 1)
        )

        self.fc_output2 = nn.Sequential(
            ResidualFC(hidden_size, fc_size1),
            ResidualFC(fc_size1, fc_size2),
            ResidualFC(fc_size2, fc_size3),
            nn.Linear(fc_size3, 1)
        )

    def forward(self, x):
        # Process through LSTM layers
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)

        # Apply attention mechanism
        attention_weights = self.attention(x).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        x_weighted = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)

        x = self.dropout(x_weighted)

        # Apply separate fully connected layers for each output
        output1 = self.fc_output1(x)
        output2 = self.fc_output2(x)

        # Concatenate the outputs
        return torch.cat((output1, output2), dim=1)
    
#Function to Convert to ONNX 
def Convert_ONNX(model): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 1, input_size, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,           # model being run
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "rnnmodel.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')

if __name__ == "__main__": 

# Initialize the model
    model = LSTMModel(input_size, hidden_size, num_stacked_layers, dropout, output_size=output_size)
    model.to(device)
    path = "/home/sj/Downloads/RNN/models/model_attention.pth"
    model.load_state_dict(torch.load(path)) 
 
    # Conversion to ONNX 
    Convert_ONNX(model=model)