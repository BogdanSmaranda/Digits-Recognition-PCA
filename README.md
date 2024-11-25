# Digits-Recognition-PCA

This project demonstrates the use of Principal Component Analysis (PCA) to reduce dimensionality and improve computational efficiency in the task of recognizing handwritten digits. A K-Nearest Neighbors (KNN) classifier is trained on the reduced dataset.

## Features
- **Dataset**: Uses the `Digits` dataset from scikit-learn, containing 1,797 handwritten digit images (0-9).
- **Dimensionality Reduction**: Applies PCA to reduce the feature space while retaining most of the variance.
- **Classification**: Implements KNN for classifying digits.
- **Evaluation**: Evaluates accuracy and visualizes the results.

## Steps
1. **Data Loading**:
   - Load and split the `Digits` dataset into training and testing sets.
2. **Preprocessing**:
   - Normalize the data using `StandardScaler`.
3. **PCA**:
   - Reduce the dimensionality of the dataset.
   - Visualize the cumulative variance explained by principal components.
4. **Classification**:
   - Train a KNN classifier on the reduced data.
   - Evaluate accuracy on the test set.
5. **Visualization**:
   - Display sample test images with predicted and actual labels.

## Requirements
Install the required Python libraries:
```bash
pip install numpy scikit-learn matplotlib
