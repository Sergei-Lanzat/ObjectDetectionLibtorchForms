# Quick Start Guide

## 5-Minute Setup (if you have dependencies installed)

### Prerequisites Check
```bash
# Check if Visual Studio is installed
where msbuild

# Check if CMake is installed
cmake --version

# Verify environment variables
echo %Torch_DIR%
echo %OpenCV_DIR%
```

### Build and Run

1. **Clone and Build**:
   ```bash
   git clone https://github.com/Sergei-Lanzat/ObjectDetectionLibtorchForms.git
   cd ObjectDetectionLibtorchForms
   
   # Build native library
   mkdir build && cd build
   cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH=%Torch_DIR%
   cmake --build . --config Release
   cd ..
   
   # Build GUI (or open .sln in Visual Studio)
   msbuild ObjectDetectionLibtorchForms.sln /p:Configuration=Release /p:Platform=x64
   ```

2. **Run**:
   ```bash
   cd x64\Release
   YoloTrainerGUI.exe
   ```

## First Training Session

### Step 1: Get Sample Dataset
Download a small dataset or use your own:
```
sample_dataset/
├── images/          # 100-1000 images (.jpg, .png)
├── labels/          # Matching .txt files
└── classes.txt      # List of class names
```

### Step 2: Configure in GUI
1. **Dataset Configuration**:
   - Training Dataset: Browse to `sample_dataset/`
   - Validation Dataset: (optional) Browse to validation folder

2. **Model Configuration**:
   - Model Type: `YOLOv5s` (fastest for testing)
   - Number of Classes: Match your `classes.txt`
   - Input Size: `640` (default)
   - Pre-trained: Uncheck for first test

3. **Training Configuration**:
   - Epochs: `10` (quick test)
   - Batch Size: `8` (adjust based on GPU memory)
   - Learning Rate: `0.01`
   - Optimizer: `SGD`
   - Use GPU: Check if available

### Step 3: Train
1. Click "Start Training"
2. Watch the convergence graph
3. Check the log panel for details
4. Wait for completion (~5-30 minutes depending on data size and hardware)

### Step 4: Export
1. After training, click "Export to ONNX"
2. Choose save location
3. Model is ready for deployment!

## Troubleshooting Quick Fixes

### "Cannot find LibTorch"
```bash
# Set environment variable
setx Torch_DIR "C:\path\to\libtorch\share\cmake\Torch"
# Restart terminal and try again
```

### "Out of Memory" Error
- Reduce batch size (try 4 or 2)
- Reduce input size (try 416 or 320)
- Close other applications

### "Cannot Open Include File"
- Check that LibTorch and OpenCV paths are correct in project properties
- Verify paths exist and are accessible
- Try using absolute paths

### Training Loss Not Decreasing
- Verify dataset format is correct
- Check class IDs in labels match classes.txt
- Try lowering learning rate (0.001)

## Common Use Cases

### Transfer Learning (Recommended)
1. Download pre-trained YOLOv5s weights
2. Check "Load Pre-trained Weights"
3. Browse to `.pt` file
4. Check "Freeze Backbone" for 5 epochs
5. Start training

### Training on Custom Objects
1. Collect 100+ images per class
2. Annotate with LabelImg or similar tool
3. Organize in YOLO format
4. Train for 100-300 epochs

### Converting Existing Annotations
1. Click "Convert Annotations"
2. Select format (COCO/VOC)
3. Choose input file/folder
4. Select output directory
5. Use converted dataset for training

## Performance Expectations

### Training Time (100 epochs on 1000 images)
- **CPU (Intel i7)**: ~8-12 hours
- **GPU (GTX 1660)**: ~2-3 hours
- **GPU (RTX 3070)**: ~1-1.5 hours
- **GPU (RTX 4090)**: ~30-45 minutes

### Model Accuracy
- **Small dataset (100 images)**: 30-50% mAP
- **Medium dataset (1000 images)**: 50-70% mAP
- **Large dataset (5000+ images)**: 70-85% mAP
- **Pre-trained + fine-tuning**: +10-20% improvement

## Next Steps

After successfully training your first model:

1. **Experiment with Hyperparameters**:
   - Try different learning rates
   - Test various batch sizes
   - Experiment with optimizers

2. **Improve Your Dataset**:
   - Add more images
   - Balance class distribution
   - Improve annotation quality
   - Add data augmentation

3. **Try Different Models**:
   - YOLOv5m (better accuracy)
   - YOLOv5l (best accuracy)
   - YOLOv8s (newer architecture)

4. **Deploy Your Model**:
   - Export to ONNX
   - Use with TensorRT for inference
   - Integrate into your application

## Getting Help

- **Documentation**: Check `docs/` folder
- **Issues**: Open GitHub issue with:
  - Error message
  - Steps to reproduce
  - System info
- **Examples**: Review `examples/` folder

## Tips for Success

✅ **DO**:
- Start with pre-trained weights
- Use validation dataset
- Monitor convergence graph
- Save checkpoints regularly
- Document your experiments

❌ **DON'T**:
- Train on tiny datasets (<50 images)
- Use learning rate > 0.1
- Ignore validation loss
- Mix different annotation formats
- Forget to normalize coordinates

## Resources

### Datasets
- **COCO**: https://cocodataset.org/
- **Open Images**: https://storage.googleapis.com/openimages/web/index.html
- **Roboflow Universe**: https://universe.roboflow.com/

### Annotation Tools
- **LabelImg**: https://github.com/heartexlabs/labelImg
- **CVAT**: https://github.com/opencv/cvat
- **Roboflow**: https://roboflow.com/

### Learning Resources
- **YOLO Papers**: Search "YOLOv5" or "YOLOv8" on arXiv
- **LibTorch Tutorial**: https://pytorch.org/cppdocs/
- **Computer Vision**: https://opencv.org/

---

**Ready to train?** Launch the GUI and start your first training session! 🚀
