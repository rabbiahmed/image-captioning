"""
This script loads and visualizes sample images and their captions from the COCO dataset
using pycocotools. It demonstrates how to access image and annotation data from URLs.
"""
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
import sys  # For sys.exit()


def load_coco_apis(data_type: str, base_dir: str):
    """
    Initializes COCO API objects for instance and caption annotations.

    Args:
        data_type (str): The dataset split (e.g., 'val2017', 'train2017').
        base_dir (str): The base directory of the project, where 'annotations' folder resides.

    Returns:
        tuple[COCO, COCO]: A tuple containing COCO objects for instances and captions.
    """
    instances_anno_file = os.path.join(base_dir, f'annotations\\instances_{data_type}.json')
    captions_anno_file = os.path.join(base_dir, f'annotations\\captions_{data_type}.json')

    # Check if annotation files exist
    if not os.path.exists(instances_anno_file):
        print(f"Error: Instance annotation file not found at {instances_anno_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(captions_anno_file):
        print(f"Error: Caption annotation file not found at {captions_anno_file}", file=sys.stderr)
        sys.exit(1)

    coco_instance = COCO(instances_anno_file)
    coco_captions = COCO(captions_anno_file)
    return coco_instance, coco_captions


def display_random_image_and_captions(coco_instance: COCO, coco_captions: COCO):
    """
    Picks a random image from the dataset, displays it, and shows its corresponding captions.

    Args:
        coco_instance (COCO): COCO API object for instance annotations.
        coco_captions (COCO): COCO API object for caption annotations.
    """
    # Get image IDs from instance annotations (ensure they have associated annotations)
    image_ids_with_anns = list(coco_instance.anns.keys())
    if not image_ids_with_anns:
        print("No image annotations found to pick a random image from.")
        return

    # Pick a random annotation ID and get its corresponding image ID
    random_ann_id = np.random.choice(image_ids_with_anns)
    img_id = coco_instance.anns[random_ann_id]['image_id']

    # Load image information
    img_info = coco_instance.loadImgs(img_id)[0]
    image_url = img_info['coco_url']

    print(f"Image ID: {img_id}")
    print(f"Image URL: {image_url}")

    try:
        image_data = io.imread(image_url)

        # Display the image
        plt.figure(figsize=(10, 8))  # Set a figure size for better display
        plt.imshow(image_data)
        plt.axis('off')
        plt.title(f"Image ID: {img_id}")
        plt.show()

        # Load and display captions for the image
        caption_ann_ids = coco_captions.getAnnIds(imgIds=img_info['id'])
        captions = coco_captions.loadAnns(caption_ann_ids)
        print("\nCaptions:")
        coco_captions.showAnns(captions)

    except Exception as e:
        print(f"Failed to load or display image/captions for Image ID {img_id} from URL {image_url}.", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        print("Please check your internet connection or the image URL.", file=sys.stderr)


if __name__ == "__main__":
    current_directory = os.getcwd()
    data_type = 'val2017'  # Specifies the COCO dataset split to use as this project uses a mini COCO dataset

    print(f"Loading COCO dataset annotations for {data_type}...")
    coco_instance_api, coco_captions_api = load_coco_apis(data_type, current_directory)
    print("COCO APIs initialized.")

    print("\n--- Plotting a Sample Image and its Captions ---")
    display_random_image_and_captions(coco_instance_api, coco_captions_api)