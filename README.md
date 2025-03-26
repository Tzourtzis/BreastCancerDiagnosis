# 📈 Breast Cancer Diagnosis Optimization with Machine Learning

This repository contains a comparative study of machine learning models and dimensionality reduction techniques for diagnosing breast cancer using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

---

## 📊 Project Overview

The project explores the use of supervised ML models to classify tumors as **malignant** or **benign**. It includes preprocessing techniques, dimensionality reduction, hyperparameter tuning, and evaluation of models based on metrics like **accuracy**, **recall**, and **F-score**.

---

## 🔧 Tools & Technologies

- Python
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- Google Colab

---

## 📚 Dataset

- **Source**: Wisconsin Diagnostic Breast Cancer (WDBC)
- **Samples**: 569 instances
- **Features**: 30 real-valued attributes (radius, texture, perimeter, etc.)
- **Target**: Diagnosis label (Malignant / Benign)

---

## ⚖️ Models Evaluated

- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Perceptron

Each model was optimized using **GridSearchCV** with **5-fold cross-validation**.

---

## 🔄 Dimensionality Reduction

- **PCA**: Preserved ~99.82% of variance with just 2 components
- **LDA**: Reduced dataset to a single component due to binary classification
- **Feature Selection**: Retained only the mean attributes to simplify the dataset

---

## ✅ Key Results

- **Logistic Regression**: Highest performance with **99.42% accuracy** and **98.41% recall** on full dataset
- **Random Forest**: Robust across all experiments, especially after PCA
- **SVM**: Strong overall accuracy and recall
- **Perceptron**: High recall, slightly weaker F-score
- **KNN**: Solid performer, but recall dropped after feature reduction

---

## 🔬 Methodology Summary

1. **EDA**: Histograms, correlation heatmaps, and pairplots for insight
2. **Preprocessing**: Scaling, encoding, stratified train-test split
3. **Modeling**: Grid search, tuning, evaluation on test set
4. **Evaluation**: Accuracy, Recall, F-score, and Confusion Matrices

---

## 🔹 Conclusions

- Logistic Regression and SVM are the most reliable models on the full dataset
- Dimensionality reduction did not consistently improve results
- Random Forest retained solid performance under all scenarios

Future work could explore ensemble models, hybrid techniques, or applying the workflow to other medical datasets.

---

## 💼 Authors

- Angelos Tzourtzis – [aggelos.tzurtzis@gmail.com](mailto:aggelos.tzurtzis@gmail.com)  
- Dimitrios Galatidis – [dimitriosgalatidis@yahoo.com](mailto:dimitriosgalatidis@yahoo.com)

MSc Students, University of Piraeus  
February 2025

---

## 🔗 References

- Scikit-learn Documentation: https://scikit-learn.org
- UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
- WHO Breast Cancer Stats 2020
- Technical Report (see PDF in repo)

---

Feel free to fork, explore, and run the notebooks to replicate or extend our experiments 🌐