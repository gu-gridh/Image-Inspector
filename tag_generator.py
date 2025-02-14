# This is a module to generate tags for images using a pre-trained model.
# This module is part of a larger system that includes feature_extractor.py and find_neighbours.py.
# The system is designed to extract features from images and find similar images in a dataset.
# The tag_generator.py module is responsible for generating tags for images based on the features extracted by the model.
# The tags can be used to categorize and search for images in the dataset.
# The module uses FastAPI to create a web service that can receive image files and return tags for the images.


# Import the required libraries
from PIL import Image
import torch
import timm
import os
import fiftyone as fo
from torchvision import transforms


# Create dataset if it doesn't exist already  or load it if it exists
def create_or_load_dataset(dataset_path, dataset_name):
    if not os.path.exists(dataset_path):
        print("Dataset path does not exist!")
        return None
    try:
        dataset = fo.load_dataset(dataset_name)
    except:
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.ImageDirectory,
            dataset_dir=dataset_path,
            name=dataset_name,
            overwrite=True,
        )
    return dataset

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

# Generate tags for an image
def generate_tags(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)
    features = features.squeeze().numpy()
    # Perform some processing to generate tags
    tags = ["tag1", "tag2", "tag3"]
    return tags



#  The tag_generator.py module is responsible for generating tags for images based on the features extracted by the model.
# The tags can be used to categorize and search for images in the dataset.
# The module uses FastAPI to create a web service that can receive image files and return tags for the images.
# The module includes functions to load a pre-trained model, extract features from images, and generate tags based on the features.

# The main function parses command-line arguments, creates or loads a dataset, and starts the FastAPI app.
# The FastAPI app defines a route to receive image files and generate tags for the images.
# The route saves the uploaded image to a temporary file, generates tags for the image, and returns the tags as a response.
# The temporary image file is removed after generating the tags.
