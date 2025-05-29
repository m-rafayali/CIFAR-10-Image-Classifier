# CIFAR-10 Image Classifier with ResNet50

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Gradio](https://img.shields.io/badge/Gradio-3.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A deep learning model that classifies images into 10 categories from the CIFAR-10 dataset, featuring a user-friendly Gradio web interface.

## Features

- ResNet50 model trained on CIFAR-10 dataset
- Gradio web interface for easy interaction
- Pre-trained model weights included (`resnet_model.pth`)
- Supports CPU inference
- Displays top 3 predictions with confidence scores
- Example images provided for quick testing
- Measures and displays prediction time

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Gradio 3.0+
- torchvision

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Cipher_ID.git
   cd Cipher_ID
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. The Gradio interface will launch:
   - Upload an image or use provided examples
   - View classification results and prediction time
   - Interface available locally at `http://127.0.0.1:7860`

## File Structure

```
Cipher_ID/
│
├── app.py                # Main application with Gradio interface
├── model.py              # Model creation and training code
├── resnet_model.pth      # Pre-trained model weights
├── class_names.txt       # CIFAR-10 class names
├── examples/             # Sample images for testing
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Model Details

- Architecture: ResNet50
- Input size: 32x32 RGB images
- Output: 10 classes (CIFAR-10 categories)
- Training: Cross-entropy loss, Adam optimizer
- Accuracy: [Your model's accuracy here]

## Customization

To retrain the model:
```python
from model import create_model
model = create_model(num_classes=10)
# Add your training loop here
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CIFAR-10 dataset creators
- PyTorch and Gradio communities
- ResNet paper authors

---
