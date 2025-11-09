# Thermal Imaging Classification

This project implements deep learning models for thermal image classification using PyTorch. The notebook provides multiple state-of-the-art model architectures and comprehensive evaluation metrics.

## Dataset

The thermal imaging dataset can be obtained from:

**DOI:** https://doi.org/10.17632/9f26y33jsx

### Dataset Setup

1. Download the dataset from the above DOI link
2. Extract the data and place it in a directory named `Processed_Data` in the project root
3. Alternatively, you can change the `data_dir` variable in the notebook's Settings section to point to your data location

```python
# In the Settings section of the notebook:
data_dir = "/path/to/your/Processed_Data"  # Update this path
```

## Project Structure

```
thermal/
├── thermal_imaging.ipynb       # Main notebook
├── Processed_Data/             # Dataset directory (download required)
├── ml_models/                  # Saved trained models
├── aug_images/                 # Augmented images cache
├── thermal_analysis_results/   # Training results and visualizations
└── README.md                   # This file
```

## Features

### Models Implemented

1. **ResNet18** - Residual Network with skip connections
2. **MobileNetV2** - Efficient mobile architecture
3. **EfficientNet** - Compound scaling network
4. **VGG16** - Classic deep convolutional network
5. **AlexNet** - Enhanced AlexNet for thermal imaging
6. **Hybrid VGG-AlexNet** - Custom hybrid model with attention mechanisms
7. **Custom CNN** - Inception-style modules for multi-scale feature extraction

### Advanced Features

- **Data Augmentation**: Intelligent augmentation with target count control
- **Mixed Precision Training**: Faster training with AMP
- **Label Smoothing**: Better generalization
- **Cosine Annealing LR**: Adaptive learning rate scheduling
- **Gradient Clipping**: Stable training
- **Attention Mechanisms**: CBAM (Convolutional Block Attention Module)
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: ROC curves, confusion matrices, training curves

### Optimization Techniques

1. AdamW optimizer with proper weight decay
2. Cosine Annealing LR scheduler
3. Label Smoothing (0.1)
4. Mixed Precision Training (AMP)
5. Gradient Clipping (max_norm=1.0)
6. Enhanced model architectures
7. Advanced regularization techniques

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch
- torchvision
- efficientnet-pytorch
- scikit-learn
- pandas
- matplotlib
- seaborn
- numpy
- Pillow

## Usage

### 1. Setup

Open the notebook and configure the settings in the "Settings" section:

```python
MODEL_DIR = "ml_models"
data_dir = "Processed_Data"  # Or your custom path
AUGMENTED_IMAGES_DIR = "aug_images"

# Control which models to train
resnet_model_training = True
mobilenet_model_training = True
efficientnet_model_training = True
vgg16_model_training = True
alexnet_model_training = True
hybrid_model_training = True
cnn_model_training = True

# Set training epochs
common_epochs = 50  # Adjust as needed
```

### 2. Run the Notebook

Execute the cells sequentially:

1. **Import Libraries**: Load all required dependencies
2. **Settings**: Configure paths and training parameters
3. **GPU Check**: Verify CUDA availability
4. **Model Definitions**: Load model architectures
5. **Data Loading**: Load and augment the dataset
6. **Training**: Train selected models
7. **Evaluation**: Generate metrics and visualizations

### 3. Results

Training results are automatically saved in `thermal_analysis_results/`:
- Training curves (CSV and PNG)
- Model evaluation metrics
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Statistical summary reports

## Model Performance Metrics

Each model is evaluated on:
- **Accuracy**: Overall classification accuracy
- **Precision**: Macro and weighted averages
- **Recall**: Macro and weighted averages
- **F1-Score**: Macro and weighted averages
- **Confusion Matrix**: Per-class performance
- **ROC AUC**: Multi-class ROC curves

## Output Files

### Training Data
- `{model_name}_training_curves_{timestamp}.csv` - Training metrics per epoch
- `enhanced_training_curves_{model_name}.png` - Training visualization

### Model Evaluation
- `model_evaluations_{timestamp}.csv` - All models comparison
- `confusion_matrix_{model_name}_{timestamp}.csv` - Confusion matrices
- `comprehensive_model_comparison.png` - Multi-metric comparison
- `enhanced_roc_curves_seaborn.png` - ROC curves
- `precision_recall_curves.png` - PR curves

### Saved Models
Trained models are saved in `ml_models/`:
- `{model_name}_model.pth` - Model weights

## Data Augmentation

The notebook includes intelligent data augmentation:
- Random affine transformations (rotation, translation, scale, shear)
- Random horizontal flips
- Color jittering (brightness, contrast, saturation)
- Target count control per class
- Augmented images caching for faster subsequent runs

## GPU Support

The notebook automatically detects and uses CUDA-enabled GPUs if available. For CPU-only training, the code will automatically fall back to CPU mode.

## Citation

If you use this dataset, please cite:

```
Dataset DOI: https://doi.org/10.17632/9f26y33jsx
```

## Troubleshooting

### Dataset Not Found
If you get a "directory not found" error, ensure:
1. The dataset is downloaded from https://doi.org/10.17632/9f26y33jsx
2. The data is placed in the `Processed_Data` directory
3. Or update the `data_dir` variable to your data location

### Out of Memory
If you encounter OOM errors:
1. Reduce `common_epochs` or batch size
2. Reduce the number of models training simultaneously
3. Use a smaller `TARGET_COUNT` for data augmentation
4. Enable only essential models

### Slow Training
To speed up training:
1. Use GPU (CUDA) if available
2. Reduce the number of epochs for initial testing
3. Use the cached augmented images (second run will be faster)
4. Reduce the target count for augmentation

## Contact

For issues or questions about this implementation, please create an issue in the repository.
