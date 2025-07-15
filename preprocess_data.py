"""
This script demonstrates how the CocoDataset and DataLoader work, including vocabulary building.
Running this script will also download images on-the-fly from COCO URLs.
"""
import os
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from pycocotools.coco import COCO
import skimage.io as io
import numpy as np

# Download NLTK punkt tokenizer data if not already present
# python -c "import nltk; nltk.download('punkt')"
# python -c "import nltk; nltk.download('punkt_tab')"
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:  # Changed from nltk.downloader.DownloadError to LookupError
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    print("NLTK 'punkt' tokenizer downloaded.")


class CocoDataset(Dataset):
    def __init__(self, annotation_file, vocab=None, transform=None, max_caption_length=50, split='train'):
        """
        Args:
            annotation_file (string): Path to the COCO captions annotation JSON file.
            vocab (dict): Pre-built vocabulary (word to index mapping). If None, it will be built.
            transform (callable, optional): Optional transform to be applied on an image.
            max_caption_length (int): Maximum length for captions. Captions longer than this will be truncated.
            split (string): 'train' or 'val' to specify which annotation file to use (e.g., captions_train2017.json).
        """
        self.annotation_file = annotation_file
        self.transform = transform
        self.max_caption_length = max_caption_length
        self.split = split

        # Initialize COCO API for captions
        self.coco_caps = COCO(annotation_file)

        # Get all image IDs from the captions annotation file
        self.image_ids = self.coco_caps.getImgIds()

        # Group captions by image ID
        self.image_captions = defaultdict(list)
        # Ensure that image_ids actually have captions associated
        # Sometimes, image IDs might be in instance annotations but not caption annotations
        # Filter out image_ids that don't have captions (though less common for caption files)
        valid_image_ids = []
        for img_id in self.image_ids:
            ann_ids = self.coco_caps.getAnnIds(imgIds=img_id)
            anns = self.coco_caps.loadAnns(ann_ids)
            if anns:  # Only if there are captions for this image
                for ann in anns:
                    self.image_captions[img_id].append(ann['caption'])
                valid_image_ids.append(img_id)
        self.image_ids = valid_image_ids  # Update image_ids to only those with captions

        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab

    def build_vocab(self, threshold=5):
        """
        Builds a vocabulary from the captions.
        Words appearing less than 'threshold' times are replaced with '<unk>'.
        """
        print("Building vocabulary...")
        word_counts = defaultdict(int)

        # Iterate through all image IDs and their associated captions
        for img_id in tqdm(self.image_ids, desc="Counting words"):
            for caption in self.image_captions[img_id]:
                tokens = word_tokenize(caption.lower())
                for token in tokens:
                    word_counts[token] += 1

        vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        idx = 4
        for word, count in word_counts.items():
            if count >= threshold:
                vocab[word] = idx
                idx += 1
        print(f"Vocabulary built with {len(vocab)} unique words.")
        return vocab

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Get image info from COCO API
        img_info = self.coco_caps.loadImgs(image_id)[0]
        image_url = img_info['coco_url']

        # Load image directly from URL
        image_np = io.imread(image_url)
        # Convert NumPy array to PIL Image for torchvision transforms
        # Ensure image_np is not empty and is a valid image before converting
        if image_np.ndim == 2:  # Handle grayscale test_images
            image = Image.fromarray(image_np, 'L').convert('RGB')
        else:
            image = Image.fromarray(image_np).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Select one random caption for training
        caption_text = self.image_captions[image_id][np.random.choice(len(self.image_captions[image_id]))]
        tokens = word_tokenize(caption_text.lower())

        # Convert tokens to indices, add <start> and <end> tokens
        caption_indices = [self.vocab["<start>"]] + \
                          [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens] + \
                          [self.vocab["<end>"]]

        # Pad or truncate caption
        if len(caption_indices) < self.max_caption_length:
            caption_indices += [self.vocab["<pad>"]] * (self.max_caption_length - len(caption_indices))
        else:
            caption_indices = caption_indices[:self.max_caption_length]

        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)

        return image, caption_tensor


def get_data_loaders(annotation_file, batch_size=32, shuffle=True, num_workers=4, vocab=None, split='train'):
    """
    Returns data loaders for the COCO dataset.
    """
    # Image transformations (example - these can be refined)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize test_images
        transforms.RandomCrop((224, 224)),  # Randomly crop for data augmentation
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    dataset = CocoDataset(annotation_file, vocab=vocab, transform=transform, split=split)

    # If vocab was built inside dataset, pass it out for consistent use in test/val sets
    if vocab is None:
        vocab = dataset.vocab

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader, vocab


if __name__ == "__main__":
    current_directory = os.getcwd()

    COCO_ANNOTATION_FILE_VAL = os.path.join(current_directory, r'annotations\captions_val2017.json')
    COCO_ANNOTATION_FILE_TRAIN = os.path.join(current_directory, r'annotations\captions_train2017.json')

    print("\n--- Demonstrating with Validation Data (small subset) ---")
    val_loader, vocab_val = get_data_loaders(
        annotation_file=COCO_ANNOTATION_FILE_VAL,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows to avoid multiprocessing issues with DataLoader
        split='val'
    )

    print(f"Number of batches in val_loader: {len(val_loader)}")
    for i, (images, captions) in enumerate(val_loader):
        print(f"Batch {i}: Images shape: {images.shape}, Captions shape: {captions.shape}")
        if i == 0:
            print("Sample caption (indices):", captions[0].tolist())
            idx_to_word = {idx: word for word, idx in vocab_val.items()}
            print("Sample caption (words):",
                  ' '.join([idx_to_word[idx.item()] for idx in captions[0] if idx.item() != vocab_val["<pad>"]]))
        break

    print("\n--- Demonstrating with Training Data (for vocabulary building) ---")
    train_loader, vocab_train = get_data_loaders(
        annotation_file=COCO_ANNOTATION_FILE_TRAIN,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows
        split='train'
    )
    print(f"Vocabulary size from training data: {len(vocab_train)}")

    print("Iterating through a few training batches (test_images will be downloaded on-the-fly):")
    for i, (images, captions) in enumerate(train_loader):
        print(f"Batch {i}: Images shape: {images.shape}, Captions shape: {captions.shape}")
        if i == 1:
            break