import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# tensor_dir = "./models/dna_hac"
# checkpoint_path = "./models/dna_hac/checkpoint.pth"
#
# state_dict = {}
#
# for filename in os.listdir(tensor_dir):
#     if filename.endswith(".tensor"):
#         file_path = os.path.join(tensor_dir, filename)
#         module = torch.jit.load(file_path, map_location="cpu")
#         tensors = module.state_dict()
#
#         for key, value in tensors.items():
#             state_dict[filename.replace(".tensor", "")] = value
#
# torch.save(state_dict, checkpoint_path)
# print("checkpoitn saved successfully")


class DoradoModel(nn.Module):
    def __init__(self):
        super(DoradoModel, self).__init__()

        # 3 Convolutional Layers
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
        )

        # LSTM Layers
        self.lstm1 = nn.LSTM(
            input_size=64, hidden_size=128, num_layers=1, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=1, batch_first=True
        )
        self.lstm3 = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=1, batch_first=True
        )

        self.fc = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = torch.relu(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        x = self.fc(x[:, -1, :])
        return x


model = DoradoModel()
state_dict = torch.load("./models/dna_hac/checkpoint.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
print("Model loaded successfully!")

test_input = torch.randn(16, 1, 100)
logits = model(test_input)
probabilities = F.softmax(logits, dim=1)

predicted_classes = torch.argmax(probabilities, dim=1)
print(predicted_classes)

base_map = ["N", "A", "C", "G", "T"]
predicted_bases = [base_map[i] for i in predicted_classes.tolist()]
print(predicted_bases)
