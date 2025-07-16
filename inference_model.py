"""
This script defines the main inference logic
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from models import EncoderCNN, DecoderRNN # Correctly imported from models.py
import os
import pickle
import nltk
from nltk.tokenize import word_tokenize

# --- Ensure NLTK punkt tokenizer is available for word_tokenize ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    print("NLTK 'punkt' tokenizer downloaded.")

# --- Main Inference Logic ---

def generate_caption(image_path, encoder_path, decoder_path, vocab_path, embed_size=256, hidden_size=512):
    """
    Generates a caption for a given image using a trained model.

    Args:
        image_path (str): Path to the input image.
        encoder_path (str): Path to the trained encoder model weights (.pth).
        decoder_path (str): Path to the trained decoder model weights (.pth).
        vocab_path (str): Path to the saved vocabulary (.pkl).
        embed_size (int): Embedding size used during training.
        hidden_size (int): Hidden size used during training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Initialize models
    # Set to eval mode is crucial for inference (disables dropout, fixes BatchNorm layers)
    encoder = EncoderCNN(embed_size).eval().to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).eval().to(device)

    # Load model weights
    # map_location ensures weights load correctly regardless of original device (CPU/GPU)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    # Image transformation (must match preprocessing during training, but for inference, use CenterCrop)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),  # Use CenterCrop for consistent inference
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension (for a single image)
    image = image.to(device)

    # --- Optimization: Explicitly disable gradient calculations for inference ---
    # This is good practice for memory efficiency and speed during inference.
    # While .eval() helps, .no_grad() explicitly ensures no gradients are computed.
    with torch.no_grad():
        # Generate features from the encoder
        features = encoder(image)

        # Generate caption using the decoder's sample method
        # The sample method itself handles the iterative generation
        sampled_ids = decoder.sample(features)

    # Convert word IDs to actual words
    idx_to_word = {idx: word for word, idx in vocab.items()}
    sampled_caption = [idx_to_word[idx] for idx in sampled_ids]

    # Join words to form the final caption, excluding <start> and <end> tokens
    final_caption = ' '.join(sampled_caption[1:-1])

    return final_caption, sampled_ids


if __name__ == "__main__":
    current_directory = os.getcwd()

    # --- Define paths to your trained model weights and vocabulary ---
    # These should match the names and directory created by train_model.py
    MODEL_SAVE_DIR = os.path.join(current_directory, 'cnn_rnn_checkpoints')

    # You need to specify the epoch for which you want to load the weights
    # For example, if train_model.py saved 'decoder-5.pth' and 'encoder-5.pth'
    EPOCH_TO_LOAD = 5  # Change this to the epoch you want to test

    ENCODER_WEIGHTS = os.path.join(MODEL_SAVE_DIR, f'encoder-{EPOCH_TO_LOAD}.pth')
    DECODER_WEIGHTS = os.path.join(MODEL_SAVE_DIR, f'decoder-{EPOCH_TO_LOAD}.pth')
    VOCAB_FILE = os.path.join(current_directory, 'vocab.pkl')

    # --- Example image for inference ---
    TEST_IMAGE_DIR = os.path.join(current_directory, 'test_images')
    if not os.path.exists(TEST_IMAGE_DIR):
        os.makedirs(TEST_IMAGE_DIR)
        print(f"Created '{TEST_IMAGE_DIR}'. Please place a test image inside this folder, e.g., 'sample_image.jpg'.")
        print("Exiting. Rerun after placing an image.")
        exit()

    SAMPLE_IMAGE_NAME = 'sample_image.jpg'  # Replace with your actual image file name
    EXAMPLE_IMAGE_PATH = os.path.join(TEST_IMAGE_DIR, SAMPLE_IMAGE_NAME)

    # Check if the example image actually exists
    if not os.path.exists(EXAMPLE_IMAGE_PATH):
        print(f"Error: Test image '{EXAMPLE_IMAGE_PATH}' not found.")
        print("Please ensure you have placed a sample image (e.g., 'sample_image.jpg') in the 'test_images' folder.")
        exit()

    # --- Generate and print caption ---
    print(f"Generating caption for: {EXAMPLE_IMAGE_PATH}")
    try:
        caption, ids = generate_caption(
            EXAMPLE_IMAGE_PATH,
            ENCODER_WEIGHTS,
            DECODER_WEIGHTS,
            VOCAB_FILE,
            embed_size=256, # Ensure these match training parameters
            hidden_size=512 # Ensure these match training parameters
        )
        print(f"\nGenerated Caption: {caption}")
        print(f"Generated IDs: {ids}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the model weights and vocabulary files exist at the specified paths.")
    except Exception as e:
        print(f"An unexpected error occurred during caption generation: {e}")