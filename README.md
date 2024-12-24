# Deep Learning for Real Waste Image Classification

## Abstract
Improper waste management poses a significant environmental challenge, necessitating innovative solutions for efficient classification and recycling. This study introduces a deep learning approach for the classification of waste images using the Real Waste Dataset[1], a publicly available dataset on Kaggle[2] comprising 4752 images categorized into 9 waste classes. Leveraging pretrained models (ResNet50[3], DenseNet121[4], MobileNetV2[5], VGG-16[6], EfficientNetV2[7], and Swin Transformer[8]) alongside a custom CNN, the proposed pipeline integrates data augmentation, class weighting, and hyperparameter optimization to address class imbalance and enhance model robustness. Among the models tested, MobileNetV2 achieves a remarkable accuracy of 93.28%, making it a highly efficient choice for resource-constrained applications. Despite these successes, challenges persist in the classification of heterogeneous categories like "Miscellaneous Trash," highlighting areas for future improvement to refine accuracy across all waste types.

## Table of Contents
- [Deep Learning for Real Waste Image Classification](#deep-learning-for-real-waste-image-classification)
  - [Abstract](#abstract)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Methodology](#methodology)
  - [Results](#results)
  - [Requirements](#requirements)
  - [Contributions](#contributions)
  - [References](#references)
  - [Authors](#authors)

## Project Overview
This project focuses on developing a robust deep learning pipeline for waste classification using the Real Waste Dataset. By leveraging state-of-the-art pretrained models and implementing strategies to address class imbalance, we aim to improve the efficiency and accuracy of waste management systems.

<div>
   <h2>Dataset</h2>
   <p>The Real Waste Dataset[1] contains 4752 labeled images spanning 9 categories:</p>
   <div style="display: flex; align-items: flex-start;">
   <ul>
      <li>Cardboard</li>
      <li>Food Organics</li>
      <li>Glass</li>
      <li>Metal</li>
      <li>Miscellaneous Trash</li>
      <li>Paper</li>
      <li>Plastic</li>
      <li>Textile Trash</li>
      <li>Vegetation</li>
   </ul>
   <div style="margin-left: 200px;">
      <img src="images/whole_dataset.png" alt="Waste categories illustration" style="max-width: 500px; height: auto;">
   </div>
   </div>
</div>

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/realwaste). The data was split into training, validation, and test sets, and augmented to boost generalization.

## Methodology
1. **Preprocessing**:
   - Dataset splitting: 80% train, 10% validation from the original 10% train set.
   - Data augmentation (Random): Rotation, vertical/horizontal flipping, zoom.
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
- **Confusion Matrix**:<br>
   <img src="images/confusion_matrices(mobilenetv2).png" alt="Confusion Matrix" style="max-width: 500px; height: auto;">
- **Precision, Recall, F1-Score**:<br>
   <img src="images/classification_metrics_trained_MobileNetV2_model.pth.png" alt="Precision, Recall, F1-Score" style="max-width: 900px; height: auto">

## Requirements
- Python 3.8 or later
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PyTorch

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


## Authors
Alessio Lani, Enrico Giordani, Lorenzo Arcioni, Marta Lombardi, Valeria Avino