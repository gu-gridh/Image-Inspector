# In this module we want to load the dataset that already has created using feature_extractor.py 
# Then find the nearest neighbours of a given image in the dataset using the features extracted from the model.

# Import the required libraries
from PIL import Image
import torch
import timm
import csv
import argparse
import fiftyone as fo
from torchvision import transforms
import fiftyone.brain as fob
from fiftyone.brain import compute_similarity

# Increase the pixel limit
Image.MAX_IMAGE_PIXELS = 1000000000  # Example: 1 billion pixels


# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and extract features using FastAPI.")
    parser.add_argument("--dataset_path", help="Path to dataset if images are stored as a dataset.")
    return parser.parse_args()


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

# Function to find nearest neighbours of a given image in the dataset
def find_neighbours(model, dataset):

    brain_key = "img_sim"

    # 🔹 Delete existing brain run if it already exists
    if brain_key in dataset.list_brain_runs():
        dataset.delete_brain_run(brain_key)
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensors = []

    for sample in dataset:
        image_path = sample.filepath
        image = Image.open(image_path).convert("RGB")  # 🔹 Convert to RGB to ensure 3 channels

        image_tensor = transform(image).unsqueeze(0)
        image_tensors.append(image_tensor)

    # Stack images for batch processing
    image_batch = torch.cat(image_tensors, dim=0)

    # Extract features
    with torch.no_grad():
        features = model(image_batch).numpy()

    # Compute similarity
    distances = compute_similarity(
        dataset,
        features=features,
        brain_key="img_sim",
        algorithm="brute_force",
        metric="cosine",
        overwrite=True
    )

    return distances

# Save the results to a CSV file
def save_results(results, output_file):
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "distance"])
        for result in results:
            writer.writerow([result["sample_id"], result["distance"]])



# Run the script and test the API endpoint using a tool like Postman or cURL. 
# You can provide the dataset name and image path as input to the endpoint, and it will return the nearest neighbors of the given image in the dataset.
# python find_neighbours.py --dataset_path dataset_name