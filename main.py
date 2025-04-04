import torch
from model import create_model ,CTCLabelConverter,OCRDataPreprocessor,recognize_text
from trainer import train_iam_huggingface
from config import CHARS
import os
from datetime import datetime
import argparse

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    params = {
        'num_epochs': 98,  # Reduced for demonstration
        'batch_size': 32,
        'learning_rate': 0.001,
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'output_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)


    model = create_model(num_classes=len(CHARS)).to(device) #Simplified

    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters())}")


    print("\nStarting training...")
    try:
        trained_model = train_iam_huggingface(
            model=model,
            num_epochs=params['num_epochs'],
            batch_size=params['batch_size'],
            device=device
        )

        final_model_path = os.path.join(output_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'params': params,  # You might not need all params
            'chars': CHARS,  # Save CHARS for consistency
        }, final_model_path)
        print(f"\nTraining completed. Final model saved to {final_model_path}")


    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        checkpoint_path = os.path.join(output_dir, 'interrupted_checkpoint.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'params': params,
            'chars': CHARS,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    except Exception as e:
        print(f"\nError during training: {e}")  # No need to re-raise

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
        main()
    elif args.mode == 'eval':
        checkpoint = torch.load(args.model_path, map_location=device)
        loaded_chars = checkpoint['chars']
        if loaded_chars != CHARS:
            print("Warning: Character set mismatch between loaded model and current settings.")
        model.load_state_dict(checkpoint['model_state_dict'])
        preprocessor = OCRDataPreprocessor()
        predicted_text = recognize_text(model, args.image_path, preprocessor, converter, device)
        print(f"Predicted text: {predicted_text}")
