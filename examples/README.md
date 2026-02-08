# Example Dataset

This directory contains sample data to help you get started with YOLO training.

## Sample Dataset Structure

```
sample_dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── labels/
│   ├── image_001.txt
│   ├── image_002.txt
│   └── ...
└── classes.txt
```

## Creating Your Own Dataset

### 1. Collect Images
- Gather images relevant to your detection task
- Recommended: 100+ images per class minimum
- More data = better results

### 2. Annotate Images
Use one of these tools:
- **LabelImg**: https://github.com/heartexlabs/labelImg
  - Easy to use
  - Supports YOLO format directly
  - Free and open source

- **CVAT**: https://github.com/opencv/cvat
  - Web-based annotation
  - Team collaboration features
  - Exports to multiple formats

- **Roboflow**: https://roboflow.com/
  - Cloud-based
  - Auto-annotation features
  - Dataset augmentation

### 3. Organize Dataset
Follow this structure:

**classes.txt** - List of class names (one per line):
```
person
vehicle
animal
```

**Label Format** - Each image has a corresponding .txt file:
```
<class_id> <x_center> <y_center> <width> <height>
```

Example (`labels/image_001.txt`):
```
0 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667
```

All coordinates are normalized (0.0 to 1.0):
- `x_center = bbox_center_x / image_width`
- `y_center = bbox_center_y / image_height`
- `width = bbox_width / image_width`
- `height = bbox_height / image_height`

### 4. Split Dataset
Create separate directories for training and validation:
```
dataset/
├── train/
│   ├── images/
│   ├── labels/
│   └── classes.txt
└── val/
    ├── images/
    ├── labels/
    └── classes.txt
```

Recommended split: 80% training, 20% validation

## Pre-trained Models

Download pre-trained YOLO weights for transfer learning:
- **YOLOv5**: https://github.com/ultralytics/yolov5/releases
- **YOLOv8**: https://github.com/ultralytics/ultralytics/releases
- **YOLO26-L**: use your `yolo26l.pt` pretrained weights when selecting the YOLO26L model type

Convert PyTorch models to LibTorch:
```python
import torch

# Load PyTorch model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Convert to TorchScript
traced = torch.jit.trace(model, torch.rand(1, 3, 640, 640))

# Save
traced.save('yolov5s.pt')
```

## Sample Training Configuration

Recommended settings for beginners:
- **Model**: YOLOv5s (fastest, good accuracy)
- **Epochs**: 100-300 (depends on dataset size)
- **Batch Size**: 16 (adjust based on GPU memory)
- **Input Size**: 640x640
- **Learning Rate**: 0.01
- **Optimizer**: SGD
- **Pre-trained**: Yes (use transfer learning)

## Tips for Good Dataset

### Image Quality
- ✅ Clear, well-lit images
- ✅ Various angles and distances
- ✅ Different backgrounds
- ❌ Blurry or low-resolution images
- ❌ Poor lighting

### Annotations
- ✅ Tight bounding boxes around objects
- ✅ All instances annotated
- ✅ Consistent class labels
- ❌ Overlapping or missing boxes
- ❌ Incorrect class assignments

### Data Balance
- ✅ Similar number of examples per class
- ✅ Variety in object sizes
- ✅ Different object positions
- ❌ One class dominating dataset
- ❌ All objects centered

## Example Datasets

### Public Datasets in YOLO Format
- **COCO**: Common objects (80 classes)
  - Download: https://cocodataset.org/
  - Requires conversion to YOLO format

- **Pascal VOC**: 20 object classes
  - Download: http://host.robots.ox.ac.uk/pascal/VOC/
  - Requires conversion to YOLO format

- **Open Images**: Large-scale dataset
  - Download: https://storage.googleapis.com/openimages/web/index.html
  - Partial downloads available

### Converting Existing Datasets
Use the built-in converter in the application:
1. Click "Convert Annotations"
2. Select source format (COCO/VOC)
3. Choose input file/directory
4. Select output directory
5. Converted dataset ready to use

## Validation Tips

Before training, validate your dataset:
1. Check classes.txt matches your label files
2. Verify all images have corresponding labels
3. Ensure coordinates are normalized (0-1 range)
4. Look for annotation errors manually
5. Test with small subset first

## Need Help?

- Check USER_GUIDE.md for detailed instructions
- Visit GitHub issues for common problems
- Review training logs for errors
