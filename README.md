# Document Classifier - Tally Invoice Classifier

A document classification tool designed to categorize Tally invoices using advanced OCR and Deep Learning models.

**Note:** This version is intended for demonstration purposes only and does not reflect the full functionality of the proprietary Tally Invoice Classifier deployed on respective websites, due to copyright restrictions.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Access the Deployed Application](#access-the-deployed-application)
- [Usage](#usage)
- [Model Details](#model-details)
- [Performance Metrics](#performance-metrics)
- [FAQ](#faq)
- [Contributing](#contributing)
- [Contributors and Contact](#contributors-and-contact)
- [Project Status](#project-status)
- [License](#license)
- [Changelog](#changelog)
- [Issues and Bug Reporting](#issues-and-bug-reporting)
- [Acknowledgements](#acknowledgements)

## Introduction

Welcome to the Tally Invoice Classifier, a robust document classification tool built using Streamlit. This application leverages OCR technology and machine learning models to classify invoices into various categories, simplifying the process of managing and organizing invoice data.

## Features

### 1. User Authentication

- **Description:** The application includes a user authentication system to ensure secure access.
- **Details:** 
  - Users are required to log in to access the application.
  - The logged-in user's name is displayed in the sidebar for confirmation.
  - Specific features and functionalities are restricted based on user roles (e.g., admin vs. non-admin users).

### 2. Upload Files of Any Type (PDF or Image)

- **Description:** Users can upload multiple files for classification, including PDFs and various image formats.
- **Details:** 
  - Supported file types include PNG, JPEG, JPG, TIFF, and PDF.
  - Multiple file uploads are allowed, enabling batch processing of invoices.
  - An option to display uploaded images within the app is available for verification.

### 3. OCR Using PyTesseract

- **Description:** The application uses Optical Character Recognition (OCR) to extract text from uploaded images and PDFs.
- **Details:** 
  - PyTesseract is utilized for OCR processing.
  - Supports extraction of text from various image formats and PDFs.
  - Provides feedback on OCR processing time for performance monitoring.

### 4. Text Preprocessing

- **Description:** Extracted text is preprocessed to prepare it for model input, ensuring accurate classification.
- **Details:** 
  - Includes steps like tokenization, removing stopwords, and normalization.
  - Ensures that the text is in a format suitable for the classification models.
  - Preprocessing is done for each uploaded document individually.

### 5. Predicting Class Using Models (BERT and TFIDF)

- **Description:** The application leverages advanced machine learning models to classify the extracted text into predefined categories.
- **Details:** 
  - Users can choose between BERT and TFIDF models for classification.
  - Models are pre-trained on a diverse dataset of invoices.
  - The application loads the selected model and tokenizer/vectorizer to make predictions.
  - Predictions include confidence scores for each classified document.

### 6. Results and Confidence Scores in a Downloadable CSV Format

- **Description:** Classified results, along with confidence scores, are displayed in the application and can be downloaded as a CSV file.
- **Details:** 
  - Results include the filename, predicted label, and confidence score.
  - Users can view results in a tabular format within the application.
  - A download button allows users to save the results as a CSV file for further analysis or record-keeping.

## Prerequisites

- Python 3.8 or higher
- Streamlit
- TensorFlow/Keras
- PyTorch
- OCR libraries (e.g., Tesseract)
- Additional Python libraries as listed in `requirements.txt`

## Getting Started

Follow these instructions to set up and run the Tally Invoice Classifier on your local machine.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/tally-invoice-classifier.git
    cd tally-invoice-classifier
    ```

2. **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download and Install Tesseract:**
    - Follow instructions from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) to install Tesseract.

5. **Download and Extract Models:**
- Download the BERT model from this [link](https://drive.google.com/file/d/1kzbamC-Rd5d_QspG6NfUIsF2wCtJZ3Rx/view?usp=sharing).
- Download the TFIDF model from this [link](https://drive.google.com/file/d/1iv5HYuUU_piVrVe6ew_oNIyRsrqf_eas/view?usp=sharing).
- Extract these zip folders and save them in `root/models` directory as `BERT/` and `TFIDF/`.

## Running the Application

1. **Start the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

2. **Access the App:**
    - Open your web browser and navigate to `http://localhost:8501`.

## Access the Deployed Application

Check out the deployed application at [Tally-Invoice-Classifier](https://huggingface.co/spaces/rakshit-ayachit/Tally-Document-Classifier).

## Usage

1. **Upload Files:**
    - Click on "Upload images or PDFs" and select the files you want to classify.

2. **Select Model:**
    - Choose the classification model from the dropdown (BERT or TFIDF).

3. **Prediction:**
    - Click the "Predict" button to start the classification process.

4. **View Results:**
    - The predicted labels and confidence scores will be displayed in a table.

5. **Download Results:**
    - Click the "Download CSV File" button to save the results.

## Model Details

### BERT
- **Description:** BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model designed to understand the context of a word in search queries.
- **Training Data:** Trained on a diverse dataset of invoices from various categories.
- **Threshold:** 0.97

### TFIDF
- **Description:** TFIDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document.
- **Training Data:** Trained on a large corpus of invoice text.
- **Threshold:** 0.85

## Performance Metrics

### BERT
- **Accuracy:** 95%
- **Precision:** 94%
- **Recall:** 96%
- **F1 Score:** 95%

### TFIDF
- **Accuracy:** 90%
- **Precision:** 89%
- **Recall:** 91%
- **F1 Score:** 90%

## FAQ

**Q1: What file formats are supported for upload?**
- A: The application supports PNG, JPEG, JPG, TIFF, and PDF formats.

**Q2: How can I clear my uploaded files?**
- A: Use the "Clear Selections" button in the sidebar to remove all uploaded files.

**Q3: What should I do if the OCR extraction is incorrect?**
- A: Ensure the quality of your images is high and the text is clear. Poor image quality can affect OCR accuracy.

**Q4: How can I contribute to this project?**
- A: Follow the steps outlined in the [Contributing](#contributing) section.

## Contributing

We welcome contributions! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## Contributors and Contact

- **Srujana Akella** -  ([@srujanaakella](https://github.com/srujanaakella)) - [srujanaakella05@gmail.com](mailto:srujanaakella05@gmail.com)
- **Adit Basak** -  ([@Aditbasak](https://github.com/Aditbasak)) - [Aditbasak55@gmail.com](mailto:Aditbasak55@gmail.com)
## Project Status

This project is currently deployed and fully functional. While it is operational, additional features and improvements may be added in the future.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### [1.0.0] - 2024-07-11
- Initial release of Tally Invoice Classifier
- Features:
    - Upload and classify invoices
    - BERT and TFIDF models
    - Download classification results

## Issues and Bug Reporting

If you encounter any issues or bugs, please open an issue on GitHub [here](https://github.com/rakshit-ayachit/tally-invoice-classifier/issues). Provide detailed information about the problem, including steps to reproduce it and any relevant screenshots.

## Acknowledgements

- [Streamlit](https://www.streamlit.io/) - The framework used for building the application.
- [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) - Machine learning libraries used for model development.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - The OCR tool used for text extraction from images.
