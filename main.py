import argparse
import os
from PIL import Image
import torch
from feature_extractor import load_model, extract_features_from_image, save_features
from find_neighbours import find_neighbours
from tag_generator import generate_tags

# Load model globally
MODEL_NAME = "vit_base_patch16_224"
model = load_model(MODEL_NAME)

def extract_features(image_path):
    """Extract features from an image."""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found!")
        return

    image = Image.open(image_path).convert("RGB")
    features = extract_features_from_image(image, model)
    save_features(image_path, features)
    print(f"✅ Features extracted and saved for {image_path}")

def find_similar_images(image_path, dataset_path):
    """Find similar images in a dataset."""
    if not os.path.exists(image_path) or not os.path.exists(dataset_path):
        print("❌ Error: Image or dataset path is invalid!")
        return

    results = find_neighbours(image_path, dataset_path)
    print(f"🔍 Similar images found: {results}")

def generate_image_tags(image_path):
    """Generate tags for an image."""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found!")
        return

    tags = generate_tags(image_path)
    print(f"🏷️ Tags for {image_path}: {tags}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images using feature extraction, similarity search, and tagging.")
    parser.add_argument("--extract", help="Extract features from an image.", metavar="IMAGE_PATH")
    parser.add_argument("--find", nargs=2, help="Find similar images: <image_path> <dataset_path>", metavar=("IMAGE_PATH", "DATASET_PATH"))
    parser.add_argument("--tag", help="Generate tags for an image.", metavar="IMAGE_PATH")
    
    args = parser.parse_args()

    if args.extract:
        extract_features(args.extract)
    elif args.find:
        find_similar_images(args.find[0], args.find[1])
    elif args.tag:
        generate_image_tags(args.tag)
    else:
        print("❌ Error: No command provided! Use --help to see available options.")
