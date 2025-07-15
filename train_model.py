"""
This script defines the CNN-RNN model, the training loop, loss function, and optimizer.
It saves the model checkpoints and vocabulary after training for later use in inference.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import EncoderCNN, DecoderRNN
from tqdm import tqdm
import os
import pickle
# Assuming preprocess_data.py is in the same directory or accessible
# Import get_data_loaders for simplified data loading
from preprocess_data import get_data_loaders


def get_lengths(captions):
    """
    Calculates the actual lengths of sequences by counting non-padding tokens.
    Assumes 0 is the padding token ID.

    Args:
        captions (torch.Tensor): A batch of padded caption tensors.

    Returns:
        list: A list of integers representing the actual lengths of each caption.
    """
    # Vectorized approach: Sums up non-zero elements along the sequence dimension.
    # This is more efficient than iterating through each caption in Python.
    return (captions != 0).sum(dim=1).tolist()


def train_model(train_loader, vocab, num_epochs=10, learning_rate=0.001, embed_size=256, hidden_size=512,
                save_path='cnn_rnn_checkpoints', log_step=100): # Changed default save_path to match common usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).to(device)

    # Loss and optimizer
    # Ignore padding token (index 0) in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Optimize parameters of the decoder and the last layers of the encoder
    # This specifically optimizes the custom linear/bn layers of the encoder,
    # as the ResNet backbone is frozen (due to torch.no_grad() in EncoderCNN.forward()).
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Create directory for saving models
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Training loop
    print("Starting training...")
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, captions) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            images = images.to(device)
            captions = captions.to(device)

            # Targets are shifted by one word compared to inputs to LSTM.
            # If LSTM input is <start> W1 W2 ... Wn, then target is W1 W2 ... Wn <end>.
            # So, input to decoder should be captions[:, :-1]
            # And target for loss should be captions[:, 1:]

            # Recalculate lengths for `input_captions` as this is what's fed to LSTM.
            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]  # Targets are shifted by one word

            lengths = get_lengths(input_captions) # Uses the optimized get_lengths function

            # Forward pass
            features = encoder(images)
            outputs = decoder(features, input_captions, lengths)

            # Reshape outputs and targets for loss calculation
            # outputs: (sum of valid input token lengths in batch) x vocab_size
            # targets: (sum of valid target token lengths in batch)
            # The `outputs` from `DecoderRNN`'s forward (after `pack_padded_sequence`) are already flattened.
            # The `targets` (which are `captions[:, 1:]`) need to be flattened similarly
            # by packing them with the same lengths and then taking the data.
            packed_targets = pack_padded_sequence(target_captions, lengths, batch_first=True, enforce_sorted=False)

            loss = criterion(outputs, packed_targets.data)

            # Backward and optimize
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if (i + 1) % log_step == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

        # Save the model checkpoints after each epoch
        # It's good practice to save based on validation performance or periodically.
        torch.save(decoder.state_dict(), os.path.join(save_path, f'decoder-{epoch + 1}.pth'))
        torch.save(encoder.state_dict(), os.path.join(save_path, f'encoder-{epoch + 1}.pth'))
        print(f"Model saved for Epoch {epoch + 1}")


if __name__ == "__main__":
    # Define paths to the COCO annotation files
    current_directory = os.getcwd()
    # Using os.path.join for robust cross-platform path construction
    COCO_ANNOTATION_FILE_TRAIN = os.path.join(current_directory, 'annotations', 'captions_train2017.json')
    # If a separate validation annotation file is used, specify it here
    # COCO_ANNOTATION_FILE_VAL = os.path.join(current_directory, 'annotations', 'captions_val2017.json')

    # Hyperparameters (you'll tune these)
    BATCH_SIZE = 64  # Increase if you have more GPU memory
    NUM_EPOCHS = 5  # Start small, then increase
    LEARNING_RATE = 0.001
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    # NUM_WORKERS = 0 is recommended for Windows to avoid multiprocessing issues with DataLoader.
    # For Linux/macOS, setting to >0 (e.g., 4 or os.cpu_count()) can significantly speed up data loading.
    NUM_WORKERS = 0
    LOG_STEP = 100  # How often to print loss

    # Create data loaders for training
    # `vocab` is built from the training set and returned.
    print("Loading training data...")
    train_loader, vocab = get_data_loaders(
        annotation_file=COCO_ANNOTATION_FILE_TRAIN,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        split='train'  # Explicitly state 'train' split
    )
    print(f"Number of batches for training: {len(train_loader)}")
    print(f"Vocabulary size: {len(vocab)}")

    # Save the vocabulary for later use in inference
    VOCAB_SAVE_PATH = 'vocab.pkl'
    with open(VOCAB_SAVE_PATH, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {VOCAB_SAVE_PATH}")

    # Define path to save model checkpoints
    MODEL_SAVE_DIR = 'cnn_rnn_checkpoints'

    # Start training
    train_model(
        train_loader,
        vocab,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        save_path=MODEL_SAVE_DIR,
        log_step=LOG_STEP
    )

    print("Training finished.")