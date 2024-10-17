# BRAIN-VS: A Hybrid VGG16-SVM Model for Brain Tumor Classification

## Introduction

Brain tumors represent one of the most fatal types of cancer globally, with an alarming mortality rate of nearly 80%, as reported by Global Cancer Statistics. In 2020 alone, 321,731 people were diagnosed with primary central nervous system (CNS) tumors. Gliomas are the most common and aggressive type of malignant brain tumor, while meningiomas and pituitary tumors are more often non-malignant.

In Indonesia, the situation is dire, with a brain tumor-related death rate reaching **92%**. This high mortality rate is worsened by limited access to radiology services and MRI equipment throughout the country, delaying diagnosis and treatment.

<p align="center">
  <img src="Project Images/Gambar 1.png" alt="Gambar 1" width="300" style="margin-right: 10px;" />
  <img src="Glor1ousFinal/Project%20Images/Gambar%202.png" alt="Gambar 2" width="300" />
</p>

## Research Gap

Current methods for diagnosing brain tumors rely heavily on MRI scans and biopsies. While effective, these techniques are:
- Time-consuming and labor-intensive.
- Prone to subjectivity and human error in the diagnostic process.

Recently, machine learning has shown promise in automating brain tumor diagnosis. However, most machine learning models still require manual feature selection, which adds complexity to the process. On the other hand, deep learning approaches offer higher accuracy but often come with significant computational requirements, making them less feasible for widespread use in under-resourced regions.

## Proposed Solution: BRAIN-VS

To address these challenges, we developed **BRAIN-VS**, a hybrid model that combines:
- **VGG16** for automatic feature extraction from MRI images.
- **Support Vector Machines (SVM)** for accurate tumor classification.

### Feature Engineering with PCA
In addition to the VGG16-SVM hybrid approach, we implemented **Principal Component Analysis (PCA)** to reduce the dimensionality of the extracted features. This compression step enhances computational efficiency by reducing the data size before classification without sacrificing performance.

### Key Advantages:
- **Automatic feature extraction**: VGG16 automates the extraction of relevant features, removing the need for manual feature selection.
- **Efficient classification**: SVM classifies the brain tumor types based on the compressed feature set, providing a balance between speed and accuracy.
- **Dimensionality reduction**: PCA reduces computational overhead, making the model faster while maintaining its predictive power.

## Dataset

We used the **Masoud, 2021** dataset, which is a combined dataset from Figshare, SARTAJ, and Br35. This dataset includes **four classes**:
1. **Meningioma**
2. **Glioma**
3. **Pituitary**
4. **No Tumor**

This dataset can be accessed via [this Kaggle link](https://www.kaggle.com/dsv/2645886).

## Results

The BRAIN-VS model achieved outstanding performance:
- **Accuracy**: 96.36%
- **Execution time**: 66 seconds (feature extraction - prediction)

These results demonstrate the model's potential for real-time clinical applications, even in resource-constrained environments.

## Conclusion

**BRAIN-VS** offers a scalable, accurate, and computationally efficient solution for brain tumor classification. By leveraging a hybrid approach with VGG16 for feature extraction, PCA for dimensionality reduction, and SVM for classification, this model overcomes the limitations of manual feature selection and the high computational costs associated with deep learning models.

---

## Getting Started

### Prerequisites

To run this project, you'll need Python 3.x and the following libraries:
- TensorFlow
- Scikit-learn
- NumPy
- OpenCV
- Matplotlib
