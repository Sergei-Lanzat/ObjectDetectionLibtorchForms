# ObjectDetectionLibtorchForms

Modern C++ WinForms application for training YOLO object detection models using LibTorch, with real-time visualization and ONNX export capabilities.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows-blue.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![LibTorch](https://img.shields.io/badge/LibTorch-2.0+-orange.svg)

## 🎯 Features

- **🖥️ Modern WinForms GUI**: Intuitive interface for training configuration and monitoring
- **🚀 Multiple YOLO Versions**: Support for YOLOv5 (s/m/l), YOLOv8 (s/m), and YOLO26-L architectures
- **🔄 Transfer Learning**: Fine-tune pre-trained models on custom datasets
- **📊 Real-time Visualization**: Live training loss convergence graphs
- **🔧 Format Conversion**: Built-in converter for COCO, Pascal VOC, and LabelImg annotations to YOLO format
- **💾 ONNX Export**: Export trained models to ONNX for deployment with TensorRT, ONNX Runtime, etc.
- **⚡ GPU Acceleration**: Optional CUDA support for faster training
- **📈 Progress Monitoring**: Detailed logging and progress tracking

## 🏗️ Architecture

The project consists of two main components:

1. **YoloTrainer** (Native C++ Library):
   - YOLO model implementation using LibTorch
   - Training pipeline with data loading and augmentation
   - Loss computation and optimization
   - Model serialization and ONNX export

2. **YoloTrainerGUI** (C++/CLI WinForms):
   - User-friendly GUI for training configuration
   - Real-time visualization of training metrics
   - Dataset management and annotation conversion
   - Model export interface

## 📋 Requirements

### Software
- **Windows 10/11** (64-bit)
- **Visual Studio 2022 or later** with:
  - Desktop development with C++
  - .NET desktop development
  - C++/CLI support
- **LibTorch 2.0+**: Download from [PyTorch](https://pytorch.org/)
- **OpenCV 4.x**: Download from [OpenCV](https://opencv.org/)
- **CMake 3.18+**: For building native library

### Hardware
- **Minimum**: Intel i5, 8 GB RAM, 10 GB storage
- **Recommended**: Intel i7/i9, 16+ GB RAM, NVIDIA GPU (GTX 1060+), 50 GB storage

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Sergei-Lanzat/ObjectDetectionLibtorchForms.git
cd ObjectDetectionLibtorchForms

# Place downloaded LibTorch in ./libtorch so CMake/VS can configure successfully (or set Torch_DIR manually)

# Set environment variables (adjust paths)
set Torch_DIR=C:\path\to\libtorch\share\cmake\Torch
set OpenCV_DIR=C:\path\to\opencv\build

# Build native library
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# Open and build GUI in Visual Studio
start ..\ObjectDetectionLibtorchForms.sln
```

For detailed build instructions, see [BUILD_INSTRUCTIONS.md](docs/BUILD_INSTRUCTIONS.md).

### 2. Prepare Dataset

Organize your dataset in YOLO format:
```
dataset/
├── images/          # Training images
├── labels/          # Annotation files
└── classes.txt      # Class names
```

Or use the built-in converter to convert from COCO/VOC format.

### 3. Train Model

1. Launch `YoloTrainerGUI.exe`
2. Configure dataset paths
3. Select model type and parameters
4. Click "Start Training"
5. Monitor progress in real-time
6. Export trained model to ONNX

For detailed usage instructions, see [USER_GUIDE.md](docs/USER_GUIDE.md).

## 📁 Project Structure

```
ObjectDetectionLibtorchForms/
├── include/                    # Header files
│   ├── YoloModel.h            # YOLO model architecture
│   ├── YoloTrainer.h          # Training pipeline
│   └── DataLoader.h           # Dataset handling
├── src/
│   ├── YoloTrainer/           # Native C++ implementation
│   │   ├── YoloModel.cpp
│   │   ├── YoloTrainer.cpp
│   │   └── DataLoader.cpp
│   └── GUI/                    # WinForms GUI
│       ├── MainForm.h
│       ├── MainForm.cpp
│       └── main.cpp
├── docs/                       # Documentation
│   ├── USER_GUIDE.md
│   └── BUILD_INSTRUCTIONS.md
├── examples/                   # Example datasets
├── CMakeLists.txt             # Build configuration
└── ObjectDetectionLibtorchForms.sln  # Visual Studio solution
```

## 🎓 Usage Example

### Training on Custom Dataset

```cpp
// Create model configuration
yolo::YoloConfig config;
config.num_classes = 3;           // Number of object classes
config.input_width = 640;
config.input_height = 640;

// Create model
auto model = std::make_shared<yolo::YoloModel>(config);

// Load pre-trained TorchScript weights (optional, uses torch::jit::load under the hood)
model->load_pretrained("yolov5s_pretrained.pt"); // or "yolo26l.pt" for YOLO26-L detection

// Configure training
yolo::TrainingConfig train_config;
train_config.num_epochs = 100;
train_config.batch_size = 16;
train_config.learning_rate = 0.01f;

// Create trainer
auto trainer = std::make_shared<yolo::YoloTrainer>(
    model, train_config, torch::kCUDA
);

// Train model
trainer->train("path/to/train", "path/to/val",
    [](const yolo::TrainingMetrics& metrics) {
        std::cout << "Epoch " << metrics.epoch 
                  << ", Loss: " << metrics.loss << std::endl;
    }
);

// Export to ONNX
model->export_onnx("trained_model.onnx");
```

## 🔧 Configuration Options

### Model Configuration
- **Model Type**: YOLOv5s, YOLOv5m, YOLOv5l, YOLOv8s, YOLOv8m
- **Input Size**: 320-1280 (multiples of 32)
- **Number of Classes**: 1-1000
- **Pre-trained Weights**: Optional for transfer learning

### Training Configuration
- **Epochs**: 1-1000
- **Batch Size**: 1-128 (depends on GPU memory)
- **Learning Rate**: 0.0001-0.1
- **Optimizer**: SGD, Adam, AdamW
- **Backbone Freezing**: Freeze initial layers for fine-tuning
- **GPU Acceleration**: Enable/disable CUDA

## 📊 Performance

Training times (approximate, YOLOv5s on COCO subset):
- **CPU (Intel i7)**: ~15 hours for 100 epochs
- **GPU (RTX 3070)**: ~2 hours for 100 epochs
- **GPU (RTX 4090)**: ~45 minutes for 100 epochs

Model accuracy (mAP@0.5) after training:
- **YOLOv5s**: 55-60% (fast, good for real-time)
- **YOLOv5m**: 60-65% (balanced)
- **YOLOv5l**: 65-70% (highest accuracy, slower)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLO architecture
- [PyTorch](https://pytorch.org/) for LibTorch
- [OpenCV](https://opencv.org/) for image processing
- YOLO research papers and community

## 📧 Contact

For questions or support:
- GitHub Issues: [Create an issue](https://github.com/Sergei-Lanzat/ObjectDetectionLibtorchForms/issues)
- Documentation: Check the `docs/` folder

## 🗺️ Roadmap

- [ ] YOLOv9 and YOLOv10 support
- [ ] Multi-GPU training
- [ ] TensorRT inference integration
- [ ] Data augmentation visualization
- [ ] Model comparison tools
- [ ] Auto hyperparameter tuning
- [ ] Web-based annotation tool integration
