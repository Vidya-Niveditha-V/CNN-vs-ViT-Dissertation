# Comparative Analysis of CNN and Vision Transformer for Breast Cancer Classification

## Overview

This project compares a convolutional neural network (CNN) and a Vision Transformer (ViT) for classifying breast mammogram images as benign or malignant. Both models use transfer learning and fine-tuning. Their performance is evaluated using classification metrics, confusion matrices, ROC curves, and Youden’s J index.

## Dataset

The study uses a publicly available Kaggle breast cancer mammogram dataset containing **3,383 images**:

| Class | Images |
|---|---:|
| Benign | 2,225 |
| Malignant | 1,158 |
| **Total** | **3,383** |

The dataset is organised into training, validation, and test directories. Images are provided at 640 × 640 pixels and resized to 224 × 224 pixels for model input.

## Models and Approach

### CNN — ResNet-50V2

- ImageNet-pretrained ResNet-50V2 backbone.
- Custom classification head with global average pooling, a dense layer, dropout, and sigmoid output.
- Data augmentation and class weighting to address class imbalance.
- Two-stage training: frozen feature extractor followed by fine-tuning.

**Implementation:** `CNN.ipynb` (R, Keras, TensorFlow)

### Vision Transformer — ViT-Base/16

- ImageNet-21k-pretrained `google/vit-base-patch16-224-in21k`.
- Two-stage training: classifier-head training followed by full-model fine-tuning.
- Image augmentation, class-weighted loss, learning-rate scheduling, and early stopping.


**Implementation:** `ViT.ipynb` (Python, PyTorch, Hugging Face Transformers)

## Evaluation

The reported test-set metrics are:

| Metric | ResNet-50V2 (CNN) | ViT-Base/16 |
|---|---:|---:|
| Accuracy | 67.9% | 68.4% |
| Precision | 56.0% | 57.5% |
| Recall (Sensitivity) | 72.7% | 65.6% |
| F1-score | 63.3% | 61.3% |
| ROC-AUC | 0.727 | 0.73 |

Youden’s J index was used to select an operating threshold from the ROC curve for each model.

## Key Findings

- ViT achieved slightly higher test accuracy, precision, and ROC-AUC.
- ResNet-50V2 achieved higher recall (sensitivity), identifying a larger proportion of malignant cases.
- Both models showed moderate performance, with signs of overfitting and limited generalisation.
- The results indicate different precision–sensitivity trade-offs rather than a clear overall winner.

## Repository Contents

```text
.
├── CNN.ipynb
├── ViT.ipynb
└── README.md
```

## Running the Notebooks

The notebooks were developed for a Kaggle environment and expect the dataset to be available at:

```text
/kaggle/input/breast-cancer-detection/
├── train/
├── valid/
└── test/
```

1. Download the dataset from its Kaggle source and add it to your Kaggle notebook.
2. Open `CNN.ipynb` in an environment with R, Keras, and TensorFlow configured.
3. Open `ViT.ipynb` in an environment with Python, PyTorch, torchvision, and Hugging Face Transformers installed.
4. Run the notebook cells in order.


## Limitations

The dataset is relatively small and class-imbalanced. Both models show only moderate test performance, and validation behaviour indicates overfitting. Results may also be affected by the available computational resources and training configuration.

## Conclusion

On this dataset, ViT achieved marginally higher accuracy and ROC-AUC, while ResNet-50V2 achieved higher sensitivity and F1-score. Further evaluation using larger, diverse, and independently sourced datasets is required. These experimental models are not clinically validated and must not be used for diagnosis or clinical decision-making.
