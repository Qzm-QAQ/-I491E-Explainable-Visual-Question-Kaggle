# -I491E-Explainable-Visual-Question-Kaggle
The details in my work
# CLEVR-X Visual Reasoning — Baseline (V2)

## 1. Introduction
This repository provides a clean and reproducible baseline solution for the **CLEVR-X Interpretable Visual Reasoning Challenge**.  
The task requires generating **both the final answer and a textual reasoning explanation** based on:

- a synthetic scene image  
- a compositional question  
- CLEVR-X ground-truth annotations  

This baseline is optimized for **Google Colab or single-GPU environments**, offering a simple pipeline that is easy to train, modify, and submit.

---

## 2. Key Features
- ✔ End-to-end multimodal training  
- ✔ Predicts **answer + rationale**  
- ✔ Compatible with low-resource environments  
- ✔ Easily customizable model components  
- ✔ Ready-to-submit inference output  
- ✔ Well-structured code for future upgrades  

---

## 3. Project Structure

```text
## 3. Project Structure

    .
    ├─ model_train/
    │  ├─ clevr_qwen2vl.py              # main training & inference script (Qwen2-VL + LoRA)
    │  ├─ CLEVR_X_training_guide.ipynb  # step-by-step training guide (optional notebook)
    │  ├─ README.md                     # documentation for this project
    │  └─ requirements.txt              # Python dependencies
    └─ (dataset is NOT stored in the repo; it is downloaded on kaggle)

---

## 4. Dataset Preparation

### 4.1 Upload all ZIP files into Colab  
Place them under `/content/` or your working directory.

### 4.2 Unzip using the verified commands
```bash
unzip clevr_x_images.zip -d data/images/
unzip clevr_x_train.zip  -d data/
unzip clevr_x_val.zip    -d data/
unzip clevr_x_test.zip   -d data/

---

## 4. Dataset Preparation

### 4.1 Upload all ZIP files into Colab  
Place them under `/content/` or your working directory.

### 4.2 Unzip using the verified commands
bash
unzip clevr_x_images.zip -d data/images/
unzip clevr_x_train.zip  -d data/
unzip clevr_x_val.zip    -d data/
unzip clevr_x_test.zip   -d data/
Images will be stored in data/images/, and metadata CSVs will be placed in data/.
```
### 5. Installation

 Install dependencies:
 ```bash 
 pip install -r requirements.txt
 ```
The environment includes:

PyTorch + TorchVision

Transformers

Pandas / NumPy

YAML config loader

## 6. Training

Run the training script:

    python train.py --config configs/default.yaml

During training, the pipeline will:

1. Load images and questions  
2. Encode questions using a text model  
3. Extract visual features using a CNN/ViT backbone  
4. Fuse visual + textual embeddings  
5. Jointly optimize:
   - Answer classification  
   - Explanation generation  

All logs, checkpoints, and validation results are automatically saved.

---

## 7. Inference & Submission

Generate predictions for the official test set:

    python infer.py --input dataset/test_non_labels.csv --output submission.csv

### Expected output format

| image_id   | answer | explanation                      |
|-----------|--------|----------------------------------|
| xxxxxxx.png| blue   | The object is blue because...    |
| 0xxxxxx.png| 2      | There are 2 spheres because...   |

The final submission file must be named:

    submission.csv

This file is directly accepted by the competition platform.

---

## 8. Baseline Model Architecture

    Image (CNN / ViT) ───────────┐
                                 ├── Fusion Layer ──→ Answer Head
    Question (Transformer/BERT) ─┘
                                 └── Explanation Generator

### Components

- Visual Encoder: CNN or ViT for image feature extraction  
- Text Encoder: Transformer/BERT for question embedding  
- Fusion Module: Simple concatenation or cross-modal interaction  
- Answer Head: Linear classifier  
- Explanation Generator: Template-based or lightweight decoder  

This V1 baseline focuses on simplicity and stability for reliable runs on limited compute.

---

## 9. Example YAML Configuration

    model:
      vision_backbone: resnet18
      text_backbone: bert-base-uncased
      fusion: concat
      hidden_dim: 512

    training:
      batch_size: 32
      lr: 2e-4
      epochs: 10
      optimizer: adamw

    dataset:
      image_path: data/images/
      train_csv: dataset/train.csv
      val_csv: dataset/val.csv

You can adjust these hyperparameters for your own experiments (e.g., larger backbones, more epochs, different learning rates).




## 10. License

MIT License

---

## 11. Contact

For questions, issues, or suggestions, feel free to open an issue or pull request.
