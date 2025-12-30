# Voice-Based Gender Recognition using MFCC and GMM

A machine learning project that identifies gender from voice recordings using Mel-Frequency Cepstral Coefficients (MFCC) and multiple classification algorithms.

## Overview

This project implements voice-based gender recognition using acoustic features and four different machine learning approaches:
- **Gaussian Mixture Models (GMM)**
- **Hidden Markov Models (HMM)**
- **Support Vector Machines (SVM)**
- **Neural Networks (NN)**

The system extracts MFCC features from audio files and trains models to classify speakers as male or female with high accuracy.

## Features

### Feature Extraction
The project uses MFCC with delta and double-delta coefficients for robust voice feature representation: [1](#0-0) 

Configuration parameters include:
- 5 MFCC coefficients
- 30 filterbank filters
- 50ms window length
- 10ms window step
- Cepstral Mean Normalization (CMN) applied

### Model Implementations

Each implementation follows the same modular structure with four main components:

1. **DataManager**: Handles dataset extraction and organization [2](#0-1) 

2. **FeaturesExtractor**: Extracts MFCC features from audio files

3. **ModelsTrainer**: Trains gender-specific models with convergence visualization [3](#0-2) 

4. **GenderIdentifier**: Performs gender classification and evaluation [4](#0-3) 

## Dataset

The project uses the **SLR45 dataset** which contains voice recordings from multiple speakers: [5](#0-4) 

The dataset includes:
- Multiple male and female speakers
- WAV format audio files
- Transcriptions for each recording
- Automatic split into training (2/3) and testing (1/3) sets

## Project Structure

```
.
├── gmmCode/          # Gaussian Mixture Model implementation
├── hmmCode/          # Hidden Markov Model implementation
├── svmCode/          # Support Vector Machine implementation
├── nnCode/           # Neural Network implementation
├── Notebook/         # Jupyter notebooks for each implementation
├── SLR45/            # Dataset directory
├── TrainingData/     # Generated training data folders
└── TestingData/      # Generated testing data folders
```

## Requirements

Based on the code, the following Python packages are required:

```
numpy
scipy
scikit-learn
python-speech-features
hmmlearn
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tano-dev/Voice-based-gender-recognition-using-MFCC-and-GMM.git
cd Voice-based-gender-recognition-using-MFCC-and-GMM
```

2. Install dependencies:
```bash
pip install numpy scipy scikit-learn python-speech-features hmmlearn matplotlib
```

3. Prepare the dataset:
```bash
python -m gmmCode.DataManager
```

This will extract and organize the SLR45 dataset into training and testing folders.

## Usage

### Training Models

Each implementation can be trained independently:

**GMM (Recommended):**
```bash
python -m gmmCode.ModelsTrainer
```

**HMM:**
```bash
python -m hmmCode.ModelsTrainer
```

**SVM:**
```bash
python -m svmCode.ModelsTrainer
```

**NN:**
```bash
python -m nnCode.ModelsTrainer
```

Training will display convergence plots showing the log-likelihood improvement over iterations. [6](#0-5) 

### Testing Models

After training, test the models:

```bash
python -m gmmCode.GenderIdentifier
```

The system will process all test files and output:
- Per-file predictions
- Confusion matrix
- Precision, recall, and F1-scores for each gender
- Overall accuracy
- Score distribution visualizations [7](#0-6) 

### Using Jupyter Notebooks

Interactive notebooks are available for each implementation:
- `Notebook/gmmCode.ipynb`
- `Notebook/hmmCode.ipynb`
- `Notebook/svmCode.ipynb`
- `Notebook/nnCode.ipynb`

## Model Details

### GMM Implementation
Uses 16-component Gaussian Mixture Models with diagonal covariance matrices: [8](#0-7) 

### Evaluation Metrics

The system provides comprehensive performance metrics:
- **Confusion Matrix**: Shows true positives, false positives, true negatives, false negatives
- **Per-Class Metrics**: Precision, recall, and F1-score for male and female classes
- **Macro-Averaged Metrics**: Overall performance across both classes
- **Score Statistics**: Distribution of model confidence scores [9](#0-8) 

### Visualization

The system generates multiple visualizations:
1. Training convergence plots (log-likelihood vs. iterations)
2. Confusion matrix heatmap
3. Accuracy comparison charts
4. Score distribution scatter plots
5. Per-class accuracy bars
6. Summary statistics panel [10](#0-9) 

## How It Works

1. **Data Preparation**: The DataManager extracts the SLR45 dataset and splits it into training/testing sets based on speaker IDs [11](#0-10) 

2. **Feature Extraction**: MFCC features are extracted from each audio file, including deltas and double-deltas

3. **Model Training**: Separate models are trained for male and female voices using the training data

4. **Gender Classification**: For a new voice sample, features are extracted and scored against both models. The higher score determines the predicted gender [12](#0-11) 

## Notes

- The GMM implementation is the primary/recommended approach as indicated by the repository name
- All implementations support the same modular architecture for easy comparison
- The system automatically handles `.wav` file filtering and error handling
- Training progress is displayed with regular updates and final convergence plots
- Models are saved as pickle files (`.gmm`, `.hmm`, `.svm`, `.nn`) for reuse
- Extensive logging and print statements provide insights into the training and testing processes
