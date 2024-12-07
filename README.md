# Dokumin Machine Learning

## Introduction

> This repository contains the machine learning model and code used for document classification in the Dokumin document management system. The model classifies images of documents into different categories, such as KTP, KK, SIM, and other document types. The model is trained using Convolutional Neural Networks (CNN) and converted to a TensorFlow Lite format for efficient use in mobile or edge devices.

## Feature
- **Image Classification**: The model can classify documents into several categories, including:
  - KTP (Indonesian Identity Card)
  - KK (Family Card)
  - SIM (Driver’s License)
  - General Documents

- **OCR and Classification**: The model helps automate document categorization as part of a larger OCR pipeline.

- **Model Formats**: The trained model is saved in both Keras (.h5) and TensorFlow Lite (.tflite) formats for various deployment options.

## Dataset
The model has been trained using a custom dataset of document images, with corresponding labels in a Pascal VOC format. The dataset consists of various categories of documents, enabling the model to classify them effectively.

The document categories in the dataset include:
- KTP (Indonesian Identity Card)
- KK (Family Card)
- SIM (Driver's License)
- General Documents

The dataset is available for download from the following link: Custom Document Classification Dataset [here](https://bit.ly/3ZwyP3Q).

## License

Dokumin API is open-source software licensed under the MIT License. See the `LICENSE` file for details.
