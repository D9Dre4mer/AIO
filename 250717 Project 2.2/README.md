# Project 2.2: Email Spam Classification

## Overview

This project implements an Email Spam Classification system using machine learning techniques. The goal is to automatically classify emails as "spam" or "not spam" (ham) based on their content. The project demonstrates data preprocessing, feature extraction, model training, evaluation, and prediction.

## Features

- Data loading and preprocessing
- Text cleaning and normalization
- Feature extraction using TF-IDF
- Model training with machine learning algorithms (e.g., Naive Bayes, Logistic Regression)
- Model evaluation with accuracy, precision, recall, and F1-score
- Prediction on new/unseen emails

## Project Structure

- `Project_2_2_spam_mail_classification.ipynb`: Main Jupyter notebook containing all code and explanations.
- `data/`: Folder for dataset files (e.g., CSV files with email data).
- `requirement.txt`: List of required Python packages.

## Installation & Usage on Google Colab

1. **Open the Notebook in Google Colab**

   - Upload `Project_2_2_spam_mail_classification.ipynb` to your Google Drive.
   - Open it with Google Colab.

2. **Install Required Packages**

   At the top of the notebook, run the following cell to install all dependencies:

   ```python
   !pip install gdown
   !pip install pandas
   !pip install seaborn
   !pip install nltk
   !pip install scikit-learn

   import nltk
   nltk.download('stopwords')
   ```

   Alternatively, you can run all the commands in `requirement.txt`:

   ```python
   !pip install -r requirement.txt
   ```

3. **Download or Upload the Dataset**

   - If the dataset is not present, follow the instructions in the notebook to download it using `gdown` or upload it manually to the Colab environment.

4. **Run the Notebook**

   - Execute each cell in order to preprocess data, train the model, and evaluate results.

## Requirements

- Python 3.x
- pandas
- seaborn
- nltk
- scikit-learn
- gdown

## License

This project is licensed under the D9Dre4mer License.