# Crack Detection in Electromechanical Materials using U-Net

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning approach for detecting and segmenting crack propagation in materials under electromechanical stress using U-Net architecture with ResNet backbones. This project applies computer vision techniques to analyze phase-field and electrical potential patterns from FEM simulations.

## 🎯 Overview

This repository implements semantic segmentation models to detect crack patterns in materials subjected to coupled mechanical and electrical stresses. The computer vision pipeline processes grayscale images from FEM simulations to identify and classify different regions:

- **Phase Field (PHI)**: Background, cracks, and holes
- **Electrical Potential (V)**: Left side, right side, and holes

## 🔬 Computer Vision Approach

### Architecture
- **U-Net** with skip connections for precise localization
- **ResNet backbones** (34, 50, 101, 152) for feature extraction
- **Transfer learning** from ImageNet pre-trained weights

### Key Features
- Multi-class semantic segmentation (3 classes)
- Pixel-level crack detection with IoU > 0.95
- Comparison between phase-field and electrical potential visualization methods
- Automated hyperparameter tuning using Keras Tuner

## 📊 Dataset

- **10,000 FEM simulations** generated using ABAQUS
- **Image dimensions**: 512×512 pixels
- **Data split**: 70% training, 20% validation, 10% test
- **Classes**: 
  - Class 0: Background
  - Class 1: Minor defects/features
  - Class 2: Major cracks/boundaries

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/crack-detection-unet.git
cd crack-detection-unet
pip install -r requirements.txt
```

### Training
```bash
# Hyperparameter tuning (recommended)
python hyper.py

# Custom U-Net with dropout
python phi.py
```

### Inference
```bash
python prediction_new_model.py
```

## 📈 Results

| Model | Backbone | IoU (PHI) | IoU (V) | F1-Score | Training Time |
|-------|----------|-----------|---------|----------|---------------|
| U-Net | ResNet34 | 0.82 | 0.96 | 0.89 | 2 hours |
| U-Net | ResNet50 | 0.74 | 0.91 | 0.83 | 3 hours |
| U-Net | ResNet152 | 0.80 | 0.94 | 0.88 | 5 hours |

**Key Finding**: Electrical potential visualization provides clearer crack path detection compared to phase-field methods.

## 🛠️ Project Structure

```
├── hyper.py                 # Hyperparameter tuning with Keras Tuner
├── phi.py                   # Custom U-Net implementation with dropout
├── prediction_new_model.py  # Inference script
├── split_data.py           # Dataset splitting utility
├── threshold_images.py     # Label generation from grayscale images
├── config.yaml             # Configuration file
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🔧 Configuration

Edit `config.yaml` to modify:
- Model architecture parameters
- Training hyperparameters
- Data paths and preprocessing settings
- Loss functions (Dice, Focal, or combined)

## 📊 Evaluation Metrics

- **IoU (Intersection over Union)**: Primary metric for segmentation accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Per-class classification accuracy

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{mavani2024crack,
  title={Detection of Crack Propagation in a Plate Using an Electromechanical Phase-Field Model and Machine Learning},
  author={Mavani, Jaykumar},
  year={2024},
  school={Leibniz Universität Hannover}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work is done in collaboration with Jaykumar Mavani
