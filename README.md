# The Image Extractor  

We are building a tool that automatically processes images from books. Our system will:  

- **Segment images** from scanned pages  
- **Generate captions** using AI  
- **Enhance image understanding** with deep learning  

Using advanced deep learning techniques, we aim to make historical and printed book images more accessible and usable.  

This project will help researchers, libraries, and digital archives extract and analyze visual content effortlessly.  

# **First Module: Image Feature Extractor**

This project extracts image features using a **Vision Transformer (ViT)** model from **Timm** and provides multiple ways to process images:

- ✅ Extract features from a single image  
- ✅ Find similar images in a dataset  
- ✅ Generate tags for images  

## **📌 Requirements**

Before running the project, make sure you have installed the dependencies:
```bash
pip install torch torchvision timm pillow fiftyone requests
```

---

## **🛠️ How to Use**

### **1️⃣ Extract Features from a Single Image**

Run the following command to extract features from an image:
```bash
python main.py --extract path/to/image.jpg
```
- Extracts features using the **Vision Transformer (ViT)** model.
- Saves the extracted features for further analysis.

---

### **2️⃣ Find Similar Images in a Dataset**

To find similar images within a dataset, run:
```bash
python main.py --find path/to/image.jpg path/to/dataset
```
- Computes image embeddings.
- Finds similar images based on feature similarity.

---

### **3️⃣ Generate Tags for an Image**

To generate tags based on an image:
```bash
python main.py --tag path/to/image.jpg
```
- Uses deep learning to assign relevant tags.

---

### **🔧 Changing the Model**

By default, the script uses `vit_base_patch16_224`. To use another model, specify it like this:
```bash
python main.py --extract path/to/image.jpg --model resnet50
```

---

### **📌 Notes**
- The extracted features **can be saved to a file** or used for further analysis.
- If the dataset or image path **does not exist**, the script will display an error.
- No FastAPI is needed; all operations run as simple Python scripts.

---

