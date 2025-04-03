import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import io
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from model import create_model, CTCLabelConverter, recognize_text,OCRDataPreprocessor
from config import CHARS
import argparse

class IAMHuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, split: str = 'train', max_width: int = 1024, max_label_length: int = None):
        self.dataset = hf_dataset[split]
        self.max_width = max_width
        self.max_label_length = max_label_length

        if self.max_label_length is not None:
          self.dataset = [item for item in self.dataset if len(item['text']) <= self.max_label_length]


    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        item = self.dataset[idx]
        if isinstance(item['image'], Image.Image):
            image = item['image'].convert('L')
        elif isinstance(item['image'], dict) and 'bytes' in item['image']:
            image = Image.open(io.BytesIO(item['image']['bytes'])).convert('L')
        else:
            raise ValueError("Unsupported image format in dataset")

        target_height = 32
        ratio = target_height / image.size[1]
        new_width = min(int(image.size[0] * ratio), self.max_width)
        image = image.resize((new_width, target_height), Image.Resampling.BILINEAR)

        image = torch.FloatTensor(np.array(image)) / 255.0
        image = (image - 0.5) / 0.5
        image = image.unsqueeze(0)

        # --- Clean the label ---
        text = item['text']
        text = text.strip()  # Remove leading/trailing whitespace
        text = ' '.join(text.split())  # Normalize multiple spaces to single spaces

        return image, text



def collate_fn(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, list[str]]:
    images, labels = zip(*batch)
    max_width = max(img.size(2) for img in images)
    padded_images = [torch.nn.functional.pad(img, (0, max_width - img.size(2), 0, 0), 'constant', 0) for img in images]
    images_tensor = torch.stack(padded_images)

    return images_tensor, labels


def train_iam_huggingface(model: nn.Module, num_epochs: int = 50, batch_size: int = 32, device: str = 'cuda'):
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("gagan3012/IAM")
    print(f"Available dataset splits: {ds.keys()}")

    # --- Inspect Sample Labels ---
    print("\n--- Sample Labels from Dataset ---")
    for i in range(5):  # Print the first 5 labels
        print(f"  Label {i}: {ds['train'][i]['text']}")
    print("--- End Sample Labels ---\n")


    validation_split = 'test' if 'test' in ds else 'train'
    if validation_split != 'test':
      print("Warning 'test' is not found, so 'train' will be used instead")

    train_dataset = IAMHuggingFaceDataset(ds, split='train', max_label_length=400) #Increased max_label_length
    val_dataset = IAMHuggingFaceDataset(ds, split=validation_split,max_label_length=400)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    converter = CTCLabelConverter(CHARS)
    criterion = nn.CTCLoss(zero_infinity=True, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch_images, batch_labels in pbar:
            batch_images = batch_images.to(device)
            label_lengths = torch.IntTensor([len(label) for label in batch_labels]).to(device)

            if (label_lengths == 0).any():
                print("Warning: Zero-length labels found in batch. Skipping this batch.")
                continue

            # --- Local Scope Verification ---
            encoded_labels = []
            for label in batch_labels:
                encoded = converter.encode(label)
                if len(encoded) != len(label):
                    print(f"Mismatched Lengths Before/After Encoding: Original='{label}', Original Len={len(label)}, Encoded Len={len(encoded)}")
                encoded_labels.append(encoded)
            batch_labels_tensor = torch.cat(encoded_labels).to(device)
            # --- End Local Scope Verification ---


            if batch_labels_tensor.size(0) != label_lengths.sum():
                print("\nError: Mismatch between concatenated label tensor size and sum of label lengths!")
                print(f"  batch_labels_tensor.shape: {batch_labels_tensor.shape}")
                print(f"  label_lengths.sum(): {label_lengths.sum()}")
                raise ValueError("Label length mismatch. Check CTCLabelConverter.encode().")


            outputs, width = model(batch_images)
            input_lengths = torch.IntTensor([width] * batch_images.size(0)).to(device)
            outputs = outputs.transpose(0, 1)

            if any(input_lengths < label_lengths):
                print("\nInput Lengths vs. Label Lengths (Error Batch):")
                for i in range(len(input_lengths)):
                    print(f"  Sample {i}: Input Length = {input_lengths[i]}, Label Length = {label_lengths[i]}")
                raise ValueError("Input seq is too short")


            try:
                loss = criterion(outputs, batch_labels_tensor, input_lengths, label_lengths)
            except RuntimeError as e:
                print(f"RuntimeError during loss calculation: {e}")
                print(f"  Max label length in batch: {max(label_lengths)}")
                print(f"  Min input length in batch: {min(input_lengths)}")
                raise

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_cer = 0.0

        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(device)
                label_lengths = torch.IntTensor([len(label) for label in batch_labels]).to(device)

                if (label_lengths == 0).any():
                    print("Warning: Zero-length labels found in validation batch. Skipping.")
                    continue

                batch_labels_encoded = [converter.encode(label) for label in batch_labels]
                batch_labels_tensor = torch.cat(batch_labels_encoded).to(device)

                outputs, width = model(batch_images)
                input_lengths = torch.IntTensor([width] * batch_images.size(0)).to(device)
                outputs = outputs.transpose(0, 1)

                loss = criterion(outputs, batch_labels_tensor, input_lengths, label_lengths)
                val_loss += loss.item()

                _, predictions = outputs.max(2)
                predictions = predictions.transpose(1, 0).contiguous().cpu()
                pred_lengths = torch.IntTensor([predictions.size(1)] * predictions.size(0))
                predicted_texts = converter.decode(predictions, pred_lengths)

                for pred, gt in zip(predicted_texts, batch_labels):
                    if pred == gt:
                        val_correct += 1
                    distance = levenshtein_distance(pred, gt)
                    val_cer += min(distance / max(len(gt), 1), 1.0)
                    val_total += 1

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        val_cer = val_cer / val_total if val_total > 0 else 0.0

        print(f'Epoch {epoch + 1}:')
        print(f'  Training Loss: {avg_train_loss:.4f}')
        print(f'  Validation Loss: {avg_val_loss:.4f}')
        print(f'  Validation Accuracy: {val_accuracy:.4f}')
        print(f'  Validation CER: {val_cer:.4f}')

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'chars': CHARS,
            }, 'best_model.pth')
    return model

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate the OCR model.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Train or evaluate.')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to model checkpoint.')
    parser.add_argument('--image_path', type=str, default='image.png', help='Path to image for inference.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use.")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    num_classes = len(CHARS)
    model = create_model(num_classes).to(device)
    converter = CTCLabelConverter(CHARS)

    if args.mode == 'train':
        train_iam_huggingface(model, num_epochs=args.num_epochs, batch_size=args.batch_size, device=device)
    elif args.mode == 'eval':
        checkpoint = torch.load(args.model_path, map_location=device)
        loaded_chars = checkpoint['chars']
        if loaded_chars != CHARS:
            print("Warning: Character set mismatch between loaded model and current settings.")
        model.load_state_dict(checkpoint['model_state_dict'])
        preprocessor = OCRDataPreprocessor()
        predicted_text = recognize_text(model, args.image_path, preprocessor, converter, device)
        print(f"Predicted text: {predicted_text}")