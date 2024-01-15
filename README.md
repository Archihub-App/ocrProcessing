# OCR processing

## About

The Archihub PDF Segmentation and OCR Plugin is designed to efficiently process PDF files, segmenting them into blocks and applying Optical Character Recognition (OCR) to specific regions.

## Features

- **PDF Segmentation:** The plugin intelligently segments PDF files into distinct blocks for efficient processing using LayoutParser.

- **OCR Integration:** OCR is selectively applied to identified blocks, extracting text for further use or analysis using Tesseract.

## Installation

1. Clone this repository to your local machine and place the downloaded folder inside the plugins folder of the application

```bash
git clone https://github.com/Archihub-App/ocrProcessing
```

2. This plugin supports multilingual OCR functionality. To enable the plugin to work with different languages, you need to download the corresponding tessdata file for the OCR from [here](https://github.com/tesseract-ocr/tessdata) and place it inside the tessdata folder inside the plugin directory.

3. Inside the models folder you should place your config_1.yaml and mymodel_1.pth files