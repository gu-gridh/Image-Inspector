# The Image Inspector  

We are building a tool that automatically processes images from books. Our system will:  

- **Segment images** from scanned pages  
- **Generate captions** using AI  
- **Enhance image understanding** with deep learning  

Using advanced deep learning techniques, we aim to make historical and printed book images more accessible and usable.  

This project will help researchers, libraries, and digital archives extract and analyze visual content effortlessly. 

# The Image Extractor  

We are building a tool that automatically processes images from books. Our system will:  

- **Segment images** from scanned pages  
- **Generate captions** using AI  
- **Enhance image understanding** with deep learning  

Using advanced deep learning techniques, we aim to make historical and printed book images more accessible and usable.  

This project will help researchers, libraries, and digital archives extract and analyze visual content effortlessly. 

# **First Module: FastAPI Image Feature Extractor**

This project extracts image features using a **Vision Transformer (ViT)** model from **Timm** and provides multiple ways to process images:

- ✅ Single image upload
- ✅ Process images from a dataset
- ✅ Process images from a CSV file

## **📌 Requirements**

Before running the project, make sure you have installed the dependencies:
```bash
pip install fastapi uvicorn torch torchvision timm pillow fiftyone requests
```

---

## **🛠️ How to Use**

### **1️⃣ Start the FastAPI Server**

Run the FastAPI server:
```bash
uvicorn feature_extractor:app --host 0.0.0.0 --port 8000
```
This will start the server at **http://127.0.0.1:8000**.

---

### **2️⃣ Extract Features from a Single Image**

Upload an image using `curl`:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/extract_features/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/image.jpg'
```
Response:
```json
{
  "features": [0.123, 0.456, 0.789, ...]
}
```

---

### **3️⃣ Process Images from a Dataset**
Run the script with a dataset path:
```bash
python feature_extractor.py --dataset_path path/to/dataset --model vit_base_patch16_224
```
- Extracts features for **all images in the dataset**.
- Stores the extracted features inside the dataset.

---

### **4️⃣ Load an Existing Dataset by Name**
If you've already created a dataset, you can load it by name instead:
```bash
python feature_extractor.py --dataset_name my-dataset --model vit_base_patch16_224
```

---

### **5️⃣ Process Images from a CSV File**
If you have a CSV file with image paths, run:
```bash
python feature_extractor.py --csv_file path/to/images.csv --model vit_base_patch16_224
```
- The script **reads the CSV file** and sends each image to the FastAPI endpoint.
- The API extracts features for each image.

---

### **🔧 Changing the Model**
By default, the script uses `vit_base_patch16_224`. To use another model, specify it like this:
```bash
python feature_extractor.py --model resnet50
```

---

### **📌 Notes**
- The extracted features **can be saved to a file** or used for further analysis.
- If the dataset or image path **does not exist**, the script will skip them.
- Make sure `uvicorn` is **running** before calling the API.

---

