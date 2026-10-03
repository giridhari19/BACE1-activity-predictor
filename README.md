# BACE Activity Prediction Using Molecular Fingerprints and Machine Learning

A machine-learning project investigating how **molecular representation** and **model architecture** affect the prediction of BACE activity.

The project compares molecular descriptors and Morgan fingerprints using Logistic Regression and Random Forest classifiers, followed by cross-validation and held-out test evaluation. A Streamlit application was developed to demonstrate the trained fingerprint-based models on new SMILES inputs.

## 🚀 Live App

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit\&logoColor=white)](https://bace1-activity-predictor.streamlit.app/)

**[Open the BACE Activity Predictor →](https://bace1-activity-predictor.streamlit.app/)**

> The application provides computational predictions from trained machine-learning models. Predictions should not be interpreted as experimental confirmation of biological activity.

---

##  Project Overview

Quantitative Structure-Activity Relationship (QSAR) modelling attempts to relate molecular structure to biological activity.

In this project, the **BACE dataset** was used as a classification problem to investigate the following question:

> **How do molecular representation and model architecture affect BACE activity classification performance?**

Two molecular representations were compared:

* **Molecular descriptors** calculated using RDKit
* **Morgan fingerprints** with radius 2 and 2048 bits

Two classification algorithms were evaluated:

* Logistic Regression
* Random Forest

This produced four model configurations:

| Molecular Representation | Model               |
| ------------------------ | ------------------- |
| Molecular descriptors    | Logistic Regression |
| Molecular descriptors    | Random Forest       |
| Morgan fingerprints      | Logistic Regression |
| Morgan fingerprints      | Random Forest       |

---

## Dataset

The project uses the **BACE dataset from MoleculeNet**.

The dataset contains molecular SMILES strings with binary activity labels corresponding to BACE activity.

### Dataset statistics

* **Total molecules:** 1,513
* **Inactive:** 822
* **Active:** 691
* **Invalid SMILES after parsing:** 0

The data was divided into:

* **80% training set**
* **20% held-out test set**

The split was stratified to preserve the approximate class distribution.

---

## Methodology

### 1. Molecular preprocessing

SMILES strings were converted into RDKit molecular objects.

Molecules that could not be parsed were identified and removed where necessary.

### 2. Molecular descriptors

An initial set of 20 RDKit descriptors was calculated.

Highly correlated descriptors were identified using pairwise Pearson correlation, and redundant descriptors were removed. Zero-variance descriptors were also removed.

The final descriptor representation contained **15 molecular descriptors**, including properties related to:

* Molecular size
* Lipophilicity
* Polarity
* Hydrogen bonding
* Molecular flexibility
* Ring systems
* Heterocycles
* Molecular complexity

### 3. Morgan fingerprints

Morgan fingerprints were generated using:

```text
Radius: 2
Fingerprint size: 2048 bits
```

These fingerprints encode local structural environments within each molecule.

### 4. Model training

Four models were trained:

```text
                    Molecular Representation
                    ┌────────────────────────┐
                    │                        │
              Descriptors              Morgan Fingerprints
                    │                        │
             ┌──────┴──────┐         ┌──────┴──────┐
             │             │         │             │
       Logistic Reg.  Random Forest  Logistic Reg. Random Forest
```

Logistic Regression was used with feature standardization for the descriptor representation.

Random Forest models were trained using 300 trees.

### 5. Model evaluation

Performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Five-fold stratified cross-validation was performed **on the training set** to assess variation across validation folds.

The held-out test set was retained for final evaluation.

---

## Cross-Validation Results

Mean ± standard deviation across five validation folds:

| Representation      | Model               |       ROC-AUC |      Accuracy |     Precision |        Recall |            F1 |
| ------------------- | ------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Descriptors         | Logistic Regression | 0.742 ± 0.040 | 0.667 ± 0.042 | 0.655 ± 0.056 | 0.581 ± 0.031 | 0.615 ± 0.040 |
| Descriptors         | Random Forest       | 0.858 ± 0.014 | 0.789 ± 0.007 | 0.775 ± 0.020 | 0.761 ± 0.028 | 0.767 ± 0.009 |
| Morgan fingerprints | Logistic Regression | 0.891 ± 0.011 | 0.809 ± 0.031 | 0.797 ± 0.040 | 0.783 ± 0.033 | 0.790 ± 0.033 |
| Morgan fingerprints | Random Forest       | 0.888 ± 0.021 | 0.819 ± 0.024 | 0.804 ± 0.030 | 0.799 ± 0.027 | 0.801 ± 0.026 |

### Key observations

* Morgan fingerprints produced substantially stronger performance than the selected molecular descriptors for both algorithms.
* Random Forest substantially improved performance over Logistic Regression when using molecular descriptors.
* Logistic Regression and Random Forest produced very similar ROC-AUC values with Morgan fingerprints.
* The difference between the two Morgan-fingerprint models was small relative to the variation across cross-validation folds.
* In this experiment, **molecular representation had a larger effect on performance than changing the model architecture**.

These observations apply specifically to this dataset, preprocessing procedure, representations, models, and evaluation setup. They should not be interpreted as universal conclusions about QSAR modelling.

---

## Held-Out Test Results

The final models were evaluated on the 20% held-out test set.

| Representation      | Model               | Accuracy | Precision | Recall |    F1 | ROC-AUC |
| ------------------- | ------------------- | -------: | --------: | -----: | ----: | ------: |
| Descriptors         | Logistic Regression |    0.634 |     0.636 |  0.457 | 0.532 |   0.730 |
| Descriptors         | Random Forest       |    0.779 |     0.793 |  0.696 | 0.741 |   0.846 |
| Morgan fingerprints | Logistic Regression |    0.779 |     0.763 |  0.746 | 0.755 |   0.888 |
| Morgan fingerprints | Random Forest       |    0.779 |     0.752 |  0.768 | 0.760 |   0.882 |

The held-out test results are reported separately from cross-validation results to distinguish final evaluation on unseen data from model validation during development.

---

## Application

The project includes a Streamlit application that allows users to enter a molecular SMILES string and obtain predictions from the two Morgan-fingerprint models.

### Application workflow

```text
SMILES input
     ↓
RDKit molecular parsing
     ↓
Morgan fingerprint generation
     ↓
┌─────────────────────┐
│ Logistic Regression │
└─────────────────────┘
          +
┌─────────────────────┐
│    Random Forest    │
└─────────────────────┘
     ↓
Predicted class
+ Active probability
+ Inactive probability
```

The application is intended as a demonstration of model deployment and should not be used as a substitute for experimental validation.

---

## 🛠️ Technologies Used

* **Python**
* **RDKit** – molecular processing, descriptors and Morgan fingerprints
* **scikit-learn** – machine-learning models and evaluation
* **Pandas** – dataset manipulation
* **NumPy** – numerical operations
* **Joblib** – model serialization
* **Streamlit** – web application and deployment
* **Git/GitHub** – version control and project hosting

---

## 📁 Repository Structure

```text
BACE-QSAR/
│
├── app.py
├── requirements.txt
│
├── lr_morgan.joblib
├── rf_morgan.joblib
│
├── README.md
│
└── docs/
    ├── methodology.md
    ├── results.md
    └── images/
```

---

## Limitations and Future Work

This project was designed as a compact investigation of molecular representation and model architecture rather than a complete production QSAR workflow.

Current limitations include:

* Evaluation was performed on a single BACE dataset.
* The main train/test evaluation uses one stratified random split.
* Scaffold-based splitting was not performed.
* Hyperparameter optimization was limited.
* No external validation dataset was used.
* The models provide computational predictions rather than experimental measurements.
* The application does not provide mechanistic interpretation of predicted activity.

Possible future extensions include:

* Scaffold-based data splitting
* External validation
* Hyperparameter optimization
* Additional molecular representations
* Additional classification algorithms
* Applicability-domain analysis
* Model interpretability
* Larger and more diverse datasets

---

## What I Learned

This project was developed to gain practical experience in applying machine learning to molecular data rather than treating QSAR as a simple model-fitting exercise.

Key areas explored include:

* Molecular data preprocessing
* RDKit molecular representations
* Descriptor selection and correlation analysis
* Morgan fingerprints
* Logistic Regression
* Random Forest
* Stratified train/test splitting
* Cross-validation
* Classification metrics
* Model comparison
* Model serialization
* Deployment of machine-learning models using Streamlit

The project also reinforced the importance of distinguishing **model performance from biological certainty** and of interpreting results within the limitations of the dataset and evaluation strategy.

---

## 📚References

1. [MoleculeNet: A Benchmark for Molecular Machine Learning](https://pubs.acs.org/doi/10.1021/acs.jmedchem.7b00794)
2. [BACE dataset – MoleculeNet](https://deepchem.readthedocs.io/en/latest/datasets/moleculenet.html)
3. [RDKit Documentation](https://www.rdkit.org/docs/)
4. [scikit-learn Documentation](https://scikit-learn.org/)
5. [Streamlit Documentation](https://docs.streamlit.io/)

---
