# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     WinForms GUI Application                     │
│                      (YoloTrainerGUI.exe)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  MainForm (C++/CLI)                                       │  │
│  │  - Dataset Configuration UI                               │  │
│  │  - Model Configuration UI                                 │  │
│  │  - Training Control UI                                    │  │
│  │  - Real-time Visualization (Charts)                       │  │
│  │  - Annotation Converter UI                                │  │
│  └──────────────┬────────────────────────────────────────────┘  │
│                 │ Calls                                          │
│                 ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Native C++ Library (YoloTrainer.lib)                    │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  YoloModel                                        │   │  │
│  │  │  - Network Architecture (CSPDarknet + PANet)     │   │  │
│  │  │  - Forward Pass                                   │   │  │
│  │  │  - Model I/O (Load/Save)                         │   │  │
│  │  │  - ONNX Export                                    │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  YoloTrainer                                      │   │  │
│  │  │  - Training Loop                                  │   │  │
│  │  │  - Loss Computation                               │   │  │
│  │  │  - Optimization (SGD/Adam/AdamW)                 │   │  │
│  │  │  - Checkpoint Management                          │   │  │
│  │  │  - Progress Callbacks                             │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  DataLoader                                       │   │  │
│  │  │  - Dataset Parsing (YOLO Format)                 │   │  │
│  │  │  - Image Loading & Preprocessing                 │   │  │
│  │  │  - Data Augmentation                              │   │  │
│  │  │  - Annotation Format Conversion                   │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  Utils                                            │   │  │
│  │  │  - File System Operations                         │   │  │
│  │  │  - String Utilities                               │   │  │
│  │  │  - Image Processing Helpers                       │   │  │
│  │  │  - Logging & Timing                               │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └──────────────┬────────────────────────────────────────────┘  │
│                 │ Uses                                           │
│                 ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  External Dependencies                                    │  │
│  │  - LibTorch (PyTorch C++ API)                            │  │
│  │  - OpenCV (Image Processing)                             │  │
│  │  - nlohmann/json (JSON Parsing)                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### 1. WinForms GUI (YoloTrainerGUI)
**Technology**: C++/CLI, .NET WinForms
**Purpose**: User interface for training configuration and monitoring

**Key Features**:
- Dataset path selection and validation
- Model configuration (type, size, classes)
- Training parameter configuration
- Real-time loss visualization with charts
- Training progress monitoring
- Annotation format conversion tools
- ONNX export interface

**Dependencies**:
- .NET Framework 4.7.2+
- System.Windows.Forms
- System.Windows.Forms.DataVisualization.Charting

### 2. YOLO Model (YoloModel.h/cpp)
**Technology**: C++17, LibTorch
**Purpose**: YOLO network architecture implementation

**Components**:
- **ConvBnSiLU**: Convolution + Batch Normalization + SiLU activation
- **C3 Block**: CSP Bottleneck with 3 convolutions
- **SPPF**: Spatial Pyramid Pooling - Fast
- **DetectionHead**: Multi-scale detection heads
- **Backbone**: CSPDarknet53
- **Neck**: PANet (Path Aggregation Network)

**Capabilities**:
- Forward pass for training and inference
- Pre-trained weight loading
- Model serialization
- ONNX export
- Backbone freezing for fine-tuning

### 3. YOLO Trainer (YoloTrainer.h/cpp)
**Technology**: C++17, LibTorch
**Purpose**: Training pipeline and optimization

**Features**:
- Configurable training loop
- Multiple optimizer support (SGD, Adam, AdamW)
- Learning rate scheduling
- Checkpoint saving/loading
- Validation during training
- Progress callbacks for GUI updates
- Early stopping support

**Loss Functions**:
- Box regression loss (IoU-based)
- Objectness loss (BCE)
- Classification loss (BCE/CrossEntropy)

### 4. Data Loader (DataLoader.h/cpp)
**Technology**: C++17, LibTorch, OpenCV
**Purpose**: Dataset handling and preprocessing

**Features**:
- YOLO format dataset parsing
- Image loading and resizing
- Normalization
- Data augmentation:
  - Random horizontal flip
  - Random scaling
  - Random crop
  - Color jittering
  - Mosaic augmentation (4-image mosaic)

**Format Converters**:
- COCO JSON to YOLO
- Pascal VOC XML to YOLO
- LabelImg XML to YOLO

### 5. Utilities (Utils.h/cpp)
**Technology**: C++17
**Purpose**: Common helper functions

**Modules**:
- **FileSystem**: Path operations, directory listing
- **StringUtils**: String manipulation
- **ImageUtils**: Image processing helpers
- **Logger**: Logging with levels
- **Timer**: Performance measurement
- **ProgressBar**: Console progress display

## Data Flow

### Training Flow
```
User Input (GUI)
    ↓
Configure Training Parameters
    ↓
Initialize Model & Trainer
    ↓
Load Dataset (DataLoader)
    ↓
Training Loop:
    ├─ Load Batch
    ├─ Augment Data
    ├─ Forward Pass (YoloModel)
    ├─ Compute Loss
    ├─ Backward Pass
    ├─ Update Weights (Optimizer)
    ├─ Update Metrics
    └─ Callback to GUI (Update Chart)
    ↓
Save Checkpoint
    ↓
Export to ONNX
```

### Dataset Format Conversion Flow
```
User Selects Source Format
    ↓
Load Annotation File
    ↓
Parse Annotations (COCO/VOC)
    ↓
Convert to YOLO Format:
    ├─ Normalize coordinates
    ├─ Map class IDs
    └─ Create label files
    ↓
Save to Output Directory
    ↓
Create classes.txt
```

## Technology Stack

### Programming Languages
- **C++17**: Core implementation
- **C++/CLI**: GUI layer bridging native and managed code

### Frameworks & Libraries
- **LibTorch 2.0+**: Deep learning framework
- **OpenCV 4.x**: Computer vision and image processing
- **.NET Framework 4.7.2+**: WinForms GUI
- **nlohmann/json**: JSON parsing (for COCO conversion)

### Build Tools
- **CMake 3.18+**: Build system for native library
- **MSBuild**: Visual Studio build system
- **Visual Studio 2022+**: IDE and compiler

## Design Patterns

### 1. Model-View-Controller (MVC)
- **Model**: YoloModel, YoloTrainer, DataLoader
- **View**: WinForms GUI (MainForm)
- **Controller**: Event handlers in MainForm

### 2. Callback Pattern
- Training progress updates via callbacks
- Decouples training logic from UI updates
- Thread-safe communication

### 3. Factory Pattern
- Model creation based on configuration
- Optimizer creation based on type selection

### 4. Strategy Pattern
- Different data augmentation strategies
- Multiple optimizer strategies

### 5. Singleton Pattern
- Logger for application-wide logging

## Threading Model

### GUI Thread
- Handles all UI events
- Updates charts and progress bars
- Receives callbacks from training thread

### Training Thread
- Runs training loop
- CPU/GPU intensive operations
- Sends progress updates via callbacks
- Can be stopped asynchronously

### Thread Safety
- Use of `InvokeRequired` for cross-thread UI updates
- Atomic flags for training control
- Mutex-protected shared state (if needed)

## Memory Management

### Native Memory
- LibTorch tensors (GPU/CPU)
- OpenCV Mat objects
- Automatic cleanup via RAII

### Managed Memory
- .NET objects (GUI components)
- Garbage collected
- Dispose pattern for resources

### Optimization
- Batch processing for efficient GPU usage
- Tensor reuse where possible
- Release mode optimizations

## Extensibility Points

### Adding New Models
1. Implement model class derived from `torch::nn::Module`
2. Register in model factory
3. Add to GUI dropdown

### Adding New Optimizers
1. Use LibTorch optimizer classes
2. Add to configuration
3. Add to GUI dropdown

### Adding New Augmentations
1. Implement in `DataAugmentation` class
2. Enable/disable via configuration
3. Optional: Add UI controls

### Adding New Format Converters
1. Implement parser for source format
2. Convert to YOLO format
3. Add to converter UI

## Performance Considerations

### Training Performance
- **Batch Size**: Larger = faster, but more memory
- **GPU**: 5-10x faster than CPU
- **Mixed Precision**: Not yet implemented (future optimization)
- **Data Loading**: Multi-threaded (configurable workers)

### Memory Usage
- **Model**: 10-100 MB depending on variant
- **Batch Data**: Batch_size × 3 × H × W × 4 bytes
- **Gradients**: Same as model parameters
- **Total**: Typically 2-8 GB for training

### Optimization Strategies
- Use smaller input size for faster training
- Reduce batch size if OOM errors
- Use pre-trained weights for faster convergence
- Enable data augmentation in separate threads

## Error Handling

### Exception Types
- **File I/O Errors**: Dataset loading, checkpoint saving
- **Memory Errors**: OOM, allocation failures
- **CUDA Errors**: GPU-related issues
- **Format Errors**: Invalid annotations

### Error Recovery
- Try-catch blocks around critical sections
- Graceful degradation (CPU fallback)
- User-friendly error messages in GUI
- Detailed logging for debugging

## Future Enhancements

### Planned Features
- [ ] Multi-GPU training support
- [ ] TensorBoard integration
- [ ] Model quantization for deployment
- [ ] Auto hyperparameter tuning
- [ ] Real-time inference preview
- [ ] Model architecture search
- [ ] Distributed training
- [ ] Web-based remote monitoring

### Optimization Opportunities
- Mixed precision training (FP16)
- Model pruning and compression
- Custom CUDA kernels for bottlenecks
- Faster data loading pipeline
- Cached dataset preprocessing
