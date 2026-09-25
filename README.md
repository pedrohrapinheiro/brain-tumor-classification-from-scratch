# brain-tumor-classification-from-scratch
Brain Tumor Classification from Scratch

Machine learning project for brain tumor classification using Logistic Regression implemented from scratch with NumPy.

The project focuses on implementing the main components of Logistic Regression manually rather than using a pre-built machine learning implementation.

Objective

Classify brain tumor images using statistical and texture features extracted from the images.

The implementation covers:

Data preprocessing
Feature extraction
Feature standardization
Logistic Regression
Sigmoid activation
Binary cross-entropy cost function
L2 regularization
Gradient computation
Gradient descent
Model prediction
Accuracy evaluation
Prediction visualization
Dataset

The dataset contains brain tumor images represented through extracted statistical and texture features.

The features used in the model are:

Mean
Variance
Standard Deviation
Entropy
Skewness
Kurtosis
Contrast
Energy
ASM
Homogeneity
Dissimilarity
Correlation
Model

The Logistic Regression model calculates:

[
z = w^T x + b
]

The sigmoid function converts the result into a probability:

[
\sigma(z) = \frac{1}{1 + e^{-z}}
]

A threshold of 0.5 is used to convert the probability into a binary classification.

prediction = (probability >= 0.5).astype(int)
Cost Function

The model uses binary cross-entropy with L2 regularization:

[
J(w,b) =
-\frac{1}{m}
\sum_{i=1}^{m}
[y_i\log(g_i)+(1-y_i)\log(1-g_i)]
+
\frac{\lambda}{2m}
\sum_{j=1}^{n}w_j^2
]

Optimization

The parameters are optimized using gradient descent:

w = w - alpha * dw
b = b - alpha * db

The gradients are calculated manually using NumPy.

Data Preprocessing

The dataset is shuffled and divided into training and validation sets.

Feature standardization is performed using statistics calculated from the training set:

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

The same training mean and standard deviation are applied to the validation set to avoid data leakage.

Results

The current implementation achieved approximately 97% validation accuracy on the dataset.

Accuracy is calculated by comparing the predicted labels with the actual labels:

accuracy = np.mean(y_pred == y_val)

Further evaluation using precision, recall, F1-score, confusion matrix, and ROC-AUC is planned.

Technologies
Python
NumPy
Pandas
Matplotlib
Jupyter Notebook
Project Structure
brain-tumor-classification-from-scratch/
│
├── data/
├── notebooks/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
Future Work
Implement confusion matrix
Add precision, recall, and F1-score
Implement ROC-AUC evaluation
Perform feature selection
Tune learning rate and regularization
Implement cross-validation
Compare with Scikit-learn
Test additional classification algorithms
Explore neural network approaches
Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used for clinical decision-making.
