
# Spam Email Classifier (Command Line Version)

A comprehensive command-line spam email classification application developed as part of the AIO2025 course. This application leverages K-Nearest Neighbors (KNN) classifier with FAISS for efficient embedding-based classification, integrated with Google Gmail API for real-time email processing.

## Features

- **Dual Classifier System**: KNN with FAISS embeddings + TF-IDF baseline for performance comparison
- **Real-time Gmail Integration**: Automatic classification and labeling of incoming emails
- **Local Email Management**: Merge and process emails from local folders
- **Comprehensive Evaluation**: Performance metrics, visualizations, and error analysis
- **Caching System**: Efficient embedding storage and reuse
- **Multi-language Support**: Uses multilingual transformer models

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Storage**: 2GB free space for models and cache

### Required Python Packages

```bash
pip install -r requirements.txt
```

## Installation

### Step 1: Download the Repository

1. Visit [GitHub Repository](https://github.com/sonvt8/AIO2025/tags)
2. Locate and download the `v2.0-command-line` tag
3. Extract the ZIP file
4. Navigate to `week8/project_spam_mails` directory
5. Move the `project_spam_mails` folder to your desired location

```bash
pip freeze > requirements.txt
```

### Step 2: Install Dependencies

```bash
# Navigate to project directory
cd project_spam_mails

# Install required packages
pip install -r requirements.txt

# If requirements.txt doesn't exist, install manually
pip install numpy pandas scikit-learn transformers faiss-cpu google-api-python-client google-auth-oauthlib nltk matplotlib seaborn
```

### Step 3: Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Step 4: Verify Installation

```bash
python main.py --help
```

## Configuration

### Google Gmail API Setup

#### 1. Enable Gmail API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Navigate to **APIs & Services** > **Library**
4. Search for "Gmail API" and click **Enable**

#### 2. Create OAuth 2.0 Credentials

1. Go to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **OAuth 2.0 Client IDs**
3. Configure consent screen if prompted:
   - User Type: External
   - Scope: Individual
4. Fill required fields and save
5. Select **Desktop app** as application type
6. Download the JSON file as `credentials.json`

#### 3. Place Credentials

```bash
# Create input directory
mkdir -p cache/input

# Move credentials file
mv credentials.json cache/input/
```

#### 4. Initial Authentication

```bash
# First run will open browser for authentication
python main.py --run-email-classifier
```

This creates `token.json` in `cache/input/` for future use.

## Usage Guide

### Initial Setup (First Run)

**Purpose**: Train the model and generate embeddings for the dataset.

```bash
python main.py
```

**What happens**:
- Loads dataset (`2cls_spam_text_cls.csv`)
- Preprocesses text data
- Generates embeddings using `intfloat/multilingual-e5-base`
- Trains KNN classifier
- Saves embeddings to `cache/embeddings/`
- Tests pipeline with sample examples

**Output**: Training logs in `logs/spam_classifier.log` and console progress

### Model Evaluation

**Purpose**: Assess model performance with metrics and visualizations.

#### Basic Evaluation
```bash
python main.py --evaluate
```

#### Custom k-values
```bash
python main.py --evaluate --k-values "1,3,7"
```

**What happens**:
- Loads precomputed embeddings
- Splits dataset into train/test sets
- Evaluates KNN for each k-value
- Compares with TF-IDF baseline
- Generates performance metrics
- Creates visualizations in `cache/output/`

**Output**:
- `cache/output/evaluation_summary.png` - 5-row visualization
- `cache/output/error_analysis.json` - Detailed metrics
- Console summary of results

### Email Management

#### Merge Local Emails

**Purpose**: Add new emails from local folders to the dataset.

```bash
python main.py --merge-emails --regenerate
```

**What happens**:
- Reads `.txt` files from `inbox/` (ham) and `spam/` (spam)
- Removes duplicates
- Appends to dataset
- Regenerates embeddings for consistency

**Note**: Always use `--regenerate` with `--merge-emails` to avoid embedding mismatch errors.

#### Real-time Gmail Classification

**Purpose**: Automatically classify and label incoming Gmail messages.

```bash
python main.py --run-email-classifier
```

**What happens**:
- Authenticates with Gmail API
- Fetches up to 10 unread emails every 30 seconds
- Classifies each email as spam/ham
- Applies labels: `Inbox_Custom` or `Spam_Custom`
- Marks emails as read
- Saves locally as `.txt` files

**Stop**: Press `Ctrl+C` to safely terminate

### Embedding Management

#### Regenerate Embeddings

**Purpose**: Update embeddings after dataset changes.

```bash
python main.py --regenerate
```

**When to use**:
- After merging emails
- After manual dataset edits
- When embedding cache is corrupted

**Note**: Can be time-consuming for large datasets.

## API Reference

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--regenerate` | flag | False | Force regeneration of embeddings |
| `--run-email-classifier` | flag | False | Enable Gmail API classification |
| `--merge-emails` | flag | False | Merge local emails into dataset |
| `--evaluate` | flag | False | Evaluate model performance |
| `--k-values` | string | "1,3,5" | Custom k-values for evaluation |

### Examples

```bash
# First run - train and generate embeddings
python main.py

# Evaluate with custom k-values
python main.py --evaluate --k-values "1,3,7"

# Merge emails and regenerate embeddings
python main.py --merge-emails --regenerate

# Run Gmail classification
python main.py --run-email-classifier

# Regenerate embeddings only
python main.py --regenerate
```

## File Structure

```
project_spam_mails/
├── cache/
│   ├── input/           # credentials.json, token.json
│   ├── output/          # plots and evaluation results
│   ├── embeddings/      # precomputed vectors
│   └── models/          # transformers & tokenizer
├── dataset/             # 2cls_spam_text_cls.csv
├── inbox/               # local ham emails (.txt files)
├── spam/                # local spam emails (.txt files)
├── logs/                # log files
├── config.py            # configuration settings
├── data_loader.py       # dataset loading utilities
├── email_handler.py     # Gmail API integration
├── embedding_generator.py # embedding generation
├── evaluator.py         # model evaluation
├── knn_classifier.py    # KNN classifier implementation
├── main.py              # main application entry point
├── spam_classifier.py   # spam classification pipeline
├── tfidf_classifier.py  # TF-IDF baseline classifier
└── README.md            # this file
```

## Troubleshooting

### Common Issues

#### 1. Seaborn Style Error
```bash
# Update matplotlib or use default style
pip install --upgrade matplotlib
```

#### 2. Missing Dataset
```bash
# Ensure dataset exists or merge emails
python main.py --merge-emails --regenerate
```

#### 3. Authentication Failed
```bash
# Check credentials and regenerate token
rm cache/input/token.json
python main.py --run-email-classifier
```

#### 4. Embedding Mismatch
```bash
# Regenerate embeddings after dataset changes
python main.py --regenerate
```

#### 5. Gmail API Issues

**No emails processed**:
- Ensure emails are unread in Gmail
- Check Gmail labels are created

**Token expired**:
```bash
rm cache/input/token.json
python main.py --run-email-classifier
```

**Invalid credentials**:
- Verify `credentials.json` is in `cache/input/`
- Check Google Cloud Console settings

### Log Monitoring

Check `logs/spam_classifier.log` for:
- Processed message IDs
- Classification results
- Error messages
- Dataset statistics

Example log entry:
```
2025-07-27 10:00:00,000 - root: Saved email ID 12345 to ./spam/email_12345.txt
```

### Performance Optimization

- **Large datasets**: Use `--regenerate` sparingly
- **Memory issues**: Close other applications
- **Slow processing**: Consider GPU acceleration with `faiss-gpu`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Version History

### v2.0 (Current)
- Added TF-IDF classifier for benchmarking
- Enhanced visualization with 5-row layout
- Improved performance optimization
- Better error handling and logging

### v1.0
- Initial KNN classifier implementation
- Basic Gmail API integration
- Command-line interface

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs in `logs/spam_classifier.log`
3. Create an issue on GitHub

## Future Development

Planned features:
- Streamlit web interface
- Real-time dashboard
- Advanced visualization tools
- Multi-language email support
- Cloud deployment options

---

## Copyright

**© 2025 Team GrID034**

This project is a product of Team GrID034, developed as part of the AIO2025 course. All rights reserved.

**Team Members:**
- [Team member names can be added here]

**Contact:** [Team contact information can be added here]

---

*This Project is maintained by Team GrID034. For questions or contributions, please contact the team.*
