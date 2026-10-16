
# AI-Powered Email Classifier

A machine-learning project that classifies email text using TF-IDF and Logistic Regression, with a Streamlit web interface.

## Features

- Email text classification
- TF-IDF feature extraction
- Logistic Regression model
- Accuracy, precision, recall, F1-score and confusion matrix evaluation
- Interactive Streamlit application

## Dataset

The training script expects a CSV file at:

`data/phishing_email.csv`

The dataset is not included in this repository. Obtain it from its original source and place it in the `data` folder before training. Verify the dataset's licence and label definitions before redistribution or interpreting predictions as phishing detection.

## Installation

Install a compatible Python version, then run:

```bash
py -m pip install -r requirements.txt
```

## Train the model

```bash
py train_model.py
```

This creates `phishing_model.joblib` locally.

## Run the application

```bash
py -m streamlit run app.py
```

## Measured results

On the current 16,498-record test split:

- Accuracy: 98.60%
- Label 0 precision: 98.62%
- Label 0 recall: 98.46%
- Label 1 precision: 98.58%
- Label 1 recall: 98.73%

Confusion matrix (actual rows, predicted columns):

```text
[[7797, 122],
 [109, 8470]]
```

These are results on the supplied dataset split. The meanings of labels 0 and 1 must be verified before interpreting these metrics as real-world phishing detection performance.

## Technologies

Python, Pandas, Scikit-learn, TF-IDF, Logistic Regression, Joblib and Streamlit.
