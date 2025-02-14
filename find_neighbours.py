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

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load default model
model = load_model()

# Find similar images
def find_neighbours(image_path, dataset):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)
    features = features.squeeze().numpy()
    distances = fob.compute_nearest_neighbors(
        dataset,
        features,
        k=5,
        distance_metric="cosine",
        field="features",
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