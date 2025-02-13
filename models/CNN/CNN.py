import os
import torch
import torch.nn as nn

class TextCNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TextCNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        conv_output_dim = 128 * ((input_dim - 2) // 2)
        self.fc1 = nn.Linear(conv_output_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def load_model(self, model_path):
        try:
            state_dict = torch.load(os.path.join('training/trained_models', model_path))
            self.load_state_dict(state_dict)
            self.eval()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, new_data):
        self.eval()
        with torch.no_grad():
            new_data_tensor = torch.tensor(new_data, dtype=torch.float32).unsqueeze(1)
            output = self(new_data_tensor)
            _, predicted = torch.max(output.data, 1)
            return predicted.item()