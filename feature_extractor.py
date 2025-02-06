from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
import torch
import timm
import io
import csv
import argparse
import os
import fiftyone as fo

# Initialize FastAPI app
app = FastAPI()

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and extract features using FastAPI.")
    parser.add_argument("--dataset_path", help="Path to dataset if images are stored as a dataset.")
    parser.add_argument("--dataset_name", help="Name of the dataset to load.")
    parser.add_argument("--input_image", help="Path to the input image.")
    parser.add_argument("--csv_file", help="Path to the CSV file containing image file paths.")
    parser.add_argument("--model", default="vit_base_patch16_224", help="The model to use for feature extraction.")
    return parser.parse_args()

# Create dataset if it doesn't exist already
def create_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        print("Dataset path does not exist!")
        return None
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageDirectory,
        dataset_dir=dataset_path,
        name="my-dataset",
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

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load default model
model = load_model()

# Save extracted features to a CSV file
def save_features(image_path, features, output_file="features.csv"):
    with open(output_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([image_path] + features)

@app.post("/extract_features/")
async def extract_features(file: UploadFile = File(...), model_name: str = "vit_base_patch16_224"):
    global model
    if model_name != model.default_cfg["architecture"]:
        model = load_model(model_name)
    
    # Read image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Extract features
    with torch.no_grad():
        features = model(image).squeeze().tolist()

    # Save to CSV
    save_features(file.filename, features)

    return {"message": f"Features saved for {file.filename}"}

# Function to process images from CSV and save features
def process_images(csv_file, model_name):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_path = row[0]
            if not os.path.exists(image_path):
                print(f"Image path {image_path} does not exist, skipping.")
                continue

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
            print(f"Processed: {image_path}")

if __name__ == "__main__":
    args = parse_arguments()

    # Process dataset if provided
    if args.dataset_path or args.dataset_name:
        dataset = create_dataset(args.dataset_path) or load_dataset(args.dataset_name)
        if dataset:
            compute_embeddings(dataset, model)

    # Process images from CSV
    elif args.csv_file:
        process_images(args.csv_file, args.model)

    # Process a single image
    elif args.input_image:
        image_path = args.input_image
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                features = model(image).squeeze().tolist()
            
            save_features(image_path, features)
            print(f"Features saved for {image_path}")
        else:
            print("Input image not found!")
