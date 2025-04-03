import torch
from model import create_model
from trainer import train_iam_huggingface
from config import CHARS
import os
from datetime import datetime

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

if __name__ == "__main__":
    main()