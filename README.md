# Transformer-based Image Captioning

## Overview
This project implements an image captioning model using a transformer-based architecture, based on the CPTR (Full Transformer Network for Image Captioning) approach with some modifications.

## Requirements
- Python 3.8.12
- Install dependencies: `pip install -r requirements.txt`
- Download Stanza English model: `stanza.download("en")`

## Dataset
Uses MS COCO 2017 dataset:
- Train images: 86,300 images
- Validation images: 18,494 images
- Test images: 18,493 images

## Setup and Running

### 1. Create Dataset
```bash
python code/create_dataset.py [ARGUMENTS]
```
Processes images, tokenizes captions, and creates vocabulary dictionary.

### 2. Train Model
```bash
python code/run_train.py [ARGUMENTS]
```
- Uses cross-entropy loss with doubly stochastic attention regularization
- Adam optimizer
- Trained for 100 epochs
- Uses pre-trained GloVe embeddings

### 3. Inference
```bash
python code/inference_test.py [ARGUMENTS]
```
- Uses beam search (size 5) for caption generation
- Generates captions with evaluation metrics (BLEU, METEOR, GLEU)

## Key Features
- Transformer encoder-decoder architecture
- Resnet101 feature extraction
- Attention visualization
- Performance analysis of generated captions

## Model Performance
- Metrics on test set (mean ± std):
  - BLEU-1: 0.7180 ± 0.17
  - BLEU-4: 0.2918 ± 0.215
  - METEOR: 0.4975 ± 0.193

## Visualization
Attention visualizations available in `images/tests` directory.

## References
1. Liu et al. (2021). CPTR: Full transformer network for image captioning.
2. Vaswani et al. (2017). Attention is all you need.

## Limitations
- Early overfitting
- Limited linguistic complexity in generated captions
- Performance varies across dataset
