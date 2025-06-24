from torchvision import transforms
from PIL import Image
import torch
import timm
import csv
import os
import fiftyone as fo
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Increase the pixel limit
Image.MAX_IMAGE_PIXELS = 1000000000  # Example: 1 billion pixels
# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset if it doesn't exist already
def create_dataset(dataset_path, dataset_name):
    if not os.path.exists(dataset_path):
        print("Dataset path does not exist!")
        return None
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageDirectory,
        dataset_dir=dataset_path,
        name=dataset_name,
        overwrite=True, #Handle error you'll get if a dataset with name already exists
    )
    return dataset

# Load dataset using FiftyOne
def load_dataset(dataset_name):
    try:
        return fo.load_dataset(dataset_name)
    except:
        print("Dataset not found. Make sure it exists.")
        return None

# Load a pre-trained model dynamically
def load_model(model_name: str = "vit_base_patch16_224"):
    model = timm.create_model(model_name, pretrained=True, num_classes=0)  # Feature extractor
    model.eval()
    return model


# Save extracted features to a CSV file
def save_features(image_path, features, output_file="features.csv"):
    with open(output_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([image_path] + features)

# Function to process images from CSV and save features
def process_images(path, model_name):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_path = row[0]
            if not os.path.exists(image_path):
                print(f"Image path {image_path} does not exist, skipping.")
                continue

            # Load default model
            model = load_model()
            # Load and transform image
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0)

            # Extract features
            with torch.no_grad():
                features = model(image).squeeze().tolist()
            
            # Save features
            save_features(image_path, features, "features_from_csv.csv")
            print(f"Processed: {image_path}")

# Compute embeddings for dataset images and save to file
def compute_embeddings(dataset, model, output_file="dataset_features.csv"):
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "features"])  # Header row

        for sample in dataset:
            image_path = sample.filepath
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                features = model(image).squeeze().tolist()
            
            writer.writerow([image_path] + features)
            # print(f"Processed: {image_path}")