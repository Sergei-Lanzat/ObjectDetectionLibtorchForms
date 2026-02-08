# YOLO Object Detection Trainer - User Guide

## Overview
This application provides a modern WinForms GUI for training YOLO (You Only Look Once) object detection models using LibTorch (C++ wrapper for PyTorch). It supports fine-tuning on custom datasets and exporting trained models to ONNX format for deployment.

## Features
- **Modern GUI**: Intuitive WinForms interface for configuring and monitoring training
- **Multiple YOLO Versions**: Support for YOLOv5 (s/m/l), YOLOv8 (s/m), and YOLO26-L architectures
- **Fine-tuning**: Load pre-trained weights and fine-tune on custom datasets
- **Data Format Conversion**: Convert COCO, Pascal VOC, or LabelImg annotations to YOLO format
- **Real-time Visualization**: Live convergence graph showing training loss
- **ONNX Export**: Export trained models to ONNX for inference with TensorRT or other frameworks
- **GPU Support**: Optional GPU acceleration for faster training

## Requirements

### Software Dependencies
- **Visual Studio 2022 or later** with C++/CLI support
- **LibTorch**: Download from https://pytorch.org/get-started/locally/
  - Choose C++/LibTorch with CUDA if using GPU
  - Extract and set environment variable `Torch_DIR` to the installation path
- **OpenCV**: Version 4.x or later
  - Download from https://opencv.org/releases/
  - Set environment variable `OpenCV_DIR` to the installation path
- **CMake**: Version 3.18 or later (for building the native library)
- **.NET Framework**: 4.7.2 or later

### Hardware Requirements
- **Minimum**:
  - CPU: Intel i5 or AMD Ryzen 5
  - RAM: 8 GB
  - Storage: 10 GB free space
- **Recommended**:
  - CPU: Intel i7/i9 or AMD Ryzen 7/9
  - RAM: 16 GB or more
  - GPU: NVIDIA GPU with CUDA support (GTX 1060 or better)
  - Storage: 50 GB free space

## Installation

### Building from Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sergei-Lanzat/ObjectDetectionLibtorchForms.git
   cd ObjectDetectionLibtorchForms
   ```

2. **Install dependencies**:
   - Download and install LibTorch
   - Download and install OpenCV
   - Set environment variables:
     ```
     set Torch_DIR=C:\path\to\libtorch
     set OpenCV_DIR=C:\path\to\opencv\build
     ```

3. **Build the native library**:
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```

4. **Build the WinForms GUI**:
   - Open `ObjectDetectionLibtorchForms.sln` in Visual Studio
   - Set `YoloTrainerGUI` as the startup project
   - Build in Release mode (x64)
   - Run the application

## Dataset Preparation

### YOLO Format
The application expects datasets in YOLO format:

```
dataset/
  ├── images/
  │   ├── train_001.jpg
  │   ├── train_002.jpg
  │   └── ...
  ├── labels/
  │   ├── train_001.txt
  │   ├── train_002.txt
  │   └── ...
  └── classes.txt
```

**classes.txt** contains class names (one per line):
```
person
car
dog
cat
```

**Label files** (one per image) contain bounding boxes:
```
<class_id> <x_center> <y_center> <width> <height>
```
Where all coordinates are normalized (0-1):
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

### Converting from Other Formats

#### From COCO Format
1. Click "Convert Annotations (COCO/VOC to YOLO)"
2. Select "COCO JSON" format
3. Browse to your COCO JSON file (e.g., `instances_train2017.json`)
4. Select output directory
5. Click "Convert"

#### From Pascal VOC Format
1. Click "Convert Annotations (COCO/VOC to YOLO)"
2. Select "Pascal VOC XML" format
3. Browse to your VOC annotations directory
4. Select output directory
5. Click "Convert"

## Training a Model

### Step 1: Configure Dataset
1. **Training Dataset Path**: Click "Browse..." and select your training dataset folder
2. **Validation Dataset Path** (optional): Select validation dataset for monitoring performance

### Step 2: Configure Model
1. **Model Type**: Choose YOLO variant (YOLOv5s recommended for beginners; select YOLO26L when using yolo26l.pt weights)
2. **Number of Classes**: Set to match your dataset
3. **Input Size**: Image resolution (640 recommended, higher = better accuracy but slower)
4. **Load Pre-trained Weights** (optional):
   - Check the box to enable fine-tuning
   - Browse to pre-trained `.pt` file
   - Recommended for better results with small datasets

### Step 3: Configure Training
1. **Epochs**: Number of training iterations (100-300 recommended)
2. **Batch Size**: Images per batch (adjust based on GPU memory)
   - 16-32 for small models (YOLOv5s)
   - 8-16 for medium models (YOLOv5m)
   - 4-8 for large models (YOLOv5l)
3. **Learning Rate**: 0.01 recommended for most cases
4. **Optimizer**: SGD (recommended) or Adam/AdamW
5. **Freeze Backbone**: Optionally freeze backbone layers for initial epochs
6. **Use GPU**: Enable if CUDA-capable GPU is available

### Step 4: Start Training
1. Click "Start Training"
2. Monitor progress:
   - **Convergence graph** shows training loss over time
   - **Log panel** displays detailed training information
   - **Progress bar** shows completion percentage
3. Training can be stopped at any time by clicking "Stop Training"

### Step 5: Export Model
1. After training completes, click "Export to ONNX"
2. Choose save location
3. The model will be saved in ONNX format for deployment

## Training Tips

### For Best Results
- **Use pre-trained weights** when possible (transfer learning)
- **Start with frozen backbone** for 5-10 epochs on small datasets
- **Use data augmentation** (enabled by default)
- **Monitor validation loss** to detect overfitting
- **Adjust learning rate** if loss doesn't decrease:
  - Too high: Loss oscillates or increases
  - Too low: Loss decreases very slowly

### Common Issues

#### Out of Memory Error
- Reduce batch size
- Reduce input size
- Use smaller model variant

#### Loss Not Decreasing
- Check dataset format is correct
- Verify class IDs match classes.txt
- Lower learning rate
- Try different optimizer

#### Model Overfitting
- Add more training data
- Use data augmentation
- Reduce model size
- Add regularization (increase weight decay)

## Model Deployment

After exporting to ONNX, the model can be used for inference with:
- **TensorRT**: NVIDIA's high-performance inference engine
- **ONNX Runtime**: Cross-platform inference
- **OpenVINO**: Intel optimization toolkit
- **Any framework supporting ONNX**: TensorFlow, PyTorch, etc.

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/Sergei-Lanzat/ObjectDetectionLibtorchForms/issues
- Documentation: Check the `docs/` folder

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- YOLO architecture by Ultralytics
- LibTorch by PyTorch team
- OpenCV computer vision library
