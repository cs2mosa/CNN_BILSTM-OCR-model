import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision import transforms
from PIL import Image

class CNNBiLSTMOCR(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 1, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(CNNBiLSTMOCR, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce height only

            nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce height only
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=128 * 2,  # Will be 128*8 with proper pooling
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_: torch.Tensor):
        out_cnn = self.CNN(input_)  # B×C×H×W
        batch_size, channels, height, width = out_cnn.size()

        # Correct reshape for sequence modeling
        out_cnn = out_cnn.permute(0, 3, 1, 2)  # B×W×C×H
        out_cnn = out_cnn.reshape(batch_size, width, channels * height)  # B×W×(C*H)

        out_lstm, _ = self.lstm(out_cnn)
        out_all = func.log_softmax(self.fc(out_lstm), dim=2)

        return out_all, width

class OCRDataPreprocessor:
    def __init__(self, height: int = 32, width: int = None):
        self.height = height
        self.width = width

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

    def process_image(self, image_path: str) -> tuple[torch.Tensor, int]:
        image = Image.open(image_path)
        if self.width is None:
            ratio = float(self.height) / image.size[1]
            new_width = int(image.size[0] * ratio)
        else:
            new_width = self.width

        image = image.resize((new_width, self.height), Image.Resampling.BILINEAR)

        image_tensor = self.transform(image)
        return image_tensor, new_width

class CTCLabelConverter:
    def __init__(self, character_set: str):
        self.character_set = ['[blank]'] + list(character_set)
        self.char2idx = {char: idx for idx, char in enumerate(self.character_set)}
        self.idx2char = {idx: char for idx, char in enumerate(self.character_set)}

    def encode(self, text: str) -> torch.LongTensor:
        encoded_text = [self.char2idx[c] for c in text if c in self.char2idx]
        return torch.LongTensor(encoded_text)


    def decode(self, text_indices: torch.Tensor, length: torch.IntTensor) -> list[str]:
        texts = []
        for indices, text_length in zip(text_indices, length):
            indices = text_indices.squeeze()
            #indices = indices[:text_length] while training
            chars = []
            previous = None
            for idx in indices:
                if idx != 0 and idx != previous:
                    chars.append(self.idx2char[int(idx)])
                previous = idx
            texts.append(''.join(chars))
        return texts


def create_model(num_classes: int, input_channels: int = 1) -> CNNBiLSTMOCR:
    model = CNNBiLSTMOCR(num_classes=num_classes, input_channels=input_channels)
    return model


def recognize_text(model: CNNBiLSTMOCR, image_path: str, preprocessor: OCRDataPreprocessor, converter: CTCLabelConverter, device: str = 'cpu') -> str:
    model.eval()
    with torch.no_grad():
        # Preprocess the image
        image_tensor, width = preprocessor.process_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        # Forward pass through the model
        outputs, _ = model(image_tensor)
        _, predictions = outputs.max(2)
        predictions = predictions.transpose(1, 0).contiguous().to(device)
        predictions_lengths = torch.IntTensor([predictions.size(1)])
        text = converter.decode(predictions, predictions_lengths)[0]

        return text