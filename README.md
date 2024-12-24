# Deep Learning for Real Waste Image Classification

## Authors
Alessio Lani, Enrico Giordani, Lorenzo Arcioni, Marta Lombardi, Valeria Avino

## Date
December 24, 2024

## Abstract
Improper waste management poses a significant environmental challenge, necessitating innovative solutions for efficient classification and recycling. This study introduces a deep learning approach for the classification of waste images using the Real Waste Dataset[1], a publicly available dataset on Kaggle[2] comprising 4752 images categorized into 9 waste classes. Leveraging pretrained models (ResNet50[3], DenseNet121[4], MobileNetV2[5], VGG-16[6], EfficientNetV2[7], and Swin Transformer[8]) alongside a custom CNN, the proposed pipeline integrates data augmentation, class weighting, and hyperparameter optimization to address class imbalance and enhance model robustness. Among the models tested, MobileNetV2 achieves a remarkable accuracy of 93.28%, making it a highly efficient choice for resource-constrained applications. Despite these successes, challenges persist in the classification of heterogeneous categories like "Miscellaneous Trash," highlighting areas for future improvement to refine accuracy across all waste types.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributions](#contributions)
- [References](#references)

## Project Overview
This project focuses on developing a robust deep learning pipeline for waste classification using the Real Waste Dataset. By leveraging state-of-the-art pretrained models and implementing strategies to address class imbalance, we aim to improve the efficiency and accuracy of waste management systems.

## Dataset
The Real Waste Dataset[1] contains 4752 labeled images spanning 9 categories:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Organic
- Miscellaneous Trash
- E-Waste
- Textile

The dataset is available on [Kaggle](https://www.kaggle.com/). The data was split into training, validation, and test sets, and augmented to address class imbalance.

## Methodology
1. **Preprocessing**:
   - Dataset splitting: 80% train, 20% validation from the original 90% train set.
   - Data augmentation: Rotation, vertical/horizontal flipping, zoom.
   - Class balancing using weighted loss functions.

2. **Models Tested**:
   - Pretrained models: ResNet50, DenseNet121, MobileNetV2, VGG-16, EfficientNetV2, Swin Transformer.
   - Custom CNN architecture.

3. **Optimization**:
   - Hyperparameter tuning.
   - Regularization techniques.

4. **Evaluation Metrics**:
   - Accuracy.
   - Confusion matrix.
   - Precision, Recall, F1-Score.

## Results
- **Best Model**: MobileNetV2
- **Accuracy**: 93.28%
- **Challenges**: Classification of heterogeneous categories like "Miscellaneous Trash."

## Setup and Installation
### Prerequisites
- Python 3.8 or later
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/waste-classification.git
   cd waste-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare the dataset by downloading it from Kaggle and placing it in the `data/` directory.
2. Train the models:
   ```bash
   python train.py --model mobilenetv2
   ```
3. Evaluate a trained model:
   ```bash
   python evaluate.py --model mobilenetv2
   ```

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request.

## References
1. Real Waste Dataset: [Kaggle Dataset](https://www.kaggle.com/)
2. ResNet50: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
3. DenseNet121: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
4. MobileNetV2: [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
5. VGG-16: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
6. EfficientNetV2: [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
7. Swin Transformer: [Swin Transformer: Hierarchical Vision Transformer](https://arxiv.org/abs/2103.14030)

---
For more details, please refer to the full project documentation.
