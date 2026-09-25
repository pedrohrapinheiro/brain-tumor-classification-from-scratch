# Brain Tumor Classification from Scratch

A machine learning project focused on implementing **Logistic Regression from scratch using NumPy** for binary classification of brain tumor images.

The project was developed to understand the mathematical foundations of Logistic Regression by implementing the model, cost function, gradients, regularization, and optimization process without relying on high-level machine learning frameworks.

---

## Overview

The pipeline transforms extracted image characteristics into a binary classification:

```text
Brain MRI Image
       │
       ▼
Feature Extraction
       │
       ▼
Statistical & Texture Features
       │
       ▼
Standardization
       │
       ▼
Logistic Regression
       │
       ▼
Binary Classification
       │
       ▼
Model Evaluation
```

## Features

The model uses statistical and texture descriptors extracted from the images:

| Feature            | Description                             |
| ------------------ | --------------------------------------- |
| Mean               | Average pixel intensity                 |
| Variance           | Pixel intensity dispersion              |
| Standard Deviation | Spread of pixel intensities             |
| Entropy            | Information content                     |
| Skewness           | Distribution asymmetry                  |
| Kurtosis           | Distribution shape                      |
| Contrast           | Local intensity variation               |
| Energy             | Texture uniformity                      |
| ASM                | Angular Second Moment                   |
| Homogeneity        | Similarity of neighboring pixels        |
| Dissimilarity      | Local pixel differences                 |
| Correlation        | Relationship between neighboring pixels |

## Logistic Regression

The model calculates the linear combination:

$$
z = w^T x + b
$$

and applies the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The resulting probability is converted into a binary prediction using a threshold of `0.5`.

```python
prediction = (probability >= 0.5).astype(int)
```

## Cost Function

The implementation uses binary cross-entropy with L2 regularization:

$$
J(w,b) =
-\frac{1}{m}
\sum_{i=1}^{m}
\left[
y_i\log(g_i) +
(1-y_i)\log(1-g_i)
\right]
+
\frac{\lambda}{2m}
\sum_{j=1}^{n}w_j^2
$$

The regularization term helps control the magnitude of the model weights.

## Optimization

The parameters are optimized using gradient descent.

```python
w = w - alpha * dw
b = b - alpha * db
```

The gradients are calculated manually using NumPy, providing a direct implementation of the optimization process.

## Data Preprocessing

The dataset is shuffled and divided into training and validation sets.

Features are standardized using the statistics calculated from the training set:

```python
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
```

The training statistics are reused for the validation set to prevent data leakage.

## Results

The current implementation achieved approximately:

**97% validation accuracy**

Accuracy is calculated by comparing predicted labels with the corresponding validation labels:

```python
accuracy = np.mean(y_pred == y_val)
```

Accuracy is only one evaluation metric. Future iterations will include additional metrics to provide a more complete assessment of the classifier.

## Technologies

```text
Python
├── NumPy
├── Pandas
└── Matplotlib

Development
└── Jupyter Notebook
```

## Project Structure

```text
brain-tumor-classification-from-scratch/
│
├── data/
│   └── dataset files
│
├── notebooks/
│   └── logistic_regression.ipynb
│
├── src/
│   └── model implementation
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Future Work

* Implement confusion matrix
* Add precision, recall, and F1-score
* Implement ROC-AUC evaluation
* Perform feature selection
* Optimize learning rate and regularization
* Implement cross-validation
* Compare the implementation with Scikit-learn
* Evaluate additional classification algorithms
* Explore neural network approaches

## Purpose

This project is primarily an educational implementation designed to strengthen understanding of:

* Supervised learning
* Binary classification
* Logistic Regression
* Gradient descent
* Regularization
* Feature engineering
* Model evaluation
* Numerical computation with NumPy

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used for clinical decision-making.
