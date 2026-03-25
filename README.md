# machine-learning

## Plant Disease Web Application (MobileNetV2 + Streamlit)

This repository now includes a simple end-to-end deep-learning web app for plant disease identification.

### 1) Install dependencies

```bash
pip install -r plant_disease_app/requirements.txt
```

### 2) Prepare dataset

Use a folder structure like:

```text
dataset/
  healthy/
    img1.jpg
    img2.jpg
  powdery_mildew/
    img3.jpg
  leaf_spot/
    img4.jpg
```

### 3) Train model (with data augmentation)

```bash
python plant_disease_app/train.py --data-dir dataset --epochs 10
```

This trains a transfer-learning model using **MobileNetV2** and saves:
- `models/plant_disease_mobilenetv2.keras`
- `models/class_names.json`

### 4) Run Streamlit app

```bash
streamlit run plant_disease_app/app.py
```

Then open the local URL shown by Streamlit, upload a leaf image, and view predicted disease class and confidence.
