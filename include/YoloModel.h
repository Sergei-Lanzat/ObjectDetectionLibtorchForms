#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>

namespace yolo {

// YOLO Detection layer output structure
struct DetectionOutput {
    torch::Tensor boxes;      // [N, 4] - x, y, w, h
    torch::Tensor scores;     // [N] - confidence scores
    torch::Tensor class_ids;  // [N] - class IDs
};

// YOLO Model configuration
struct YoloConfig {
    int num_classes = 80;
    int input_width = 640;
    int input_height = 640;
    std::vector<int> anchors;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int max_detections = 300;
    std::string model_type = "yolov5s"; // yolov5s, yolov5m, yolov5l, yolov8s, etc.
};

// Conv + BatchNorm + SiLU activation block
class ConvBnSiLU : public torch::nn::Module {
public:
    ConvBnSiLU(int in_channels, int out_channels, int kernel_size = 1, int stride = 1, int padding = 0);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::SiLU silu{nullptr};
};

// C3 Block (CSPBottleneck with 3 convolutions)
class C3 : public torch::nn::Module {
public:
    C3(int in_channels, int out_channels, int num_blocks = 1, bool shortcut = true);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential cv1{nullptr};
    torch::nn::Sequential cv2{nullptr};
    torch::nn::Sequential cv3{nullptr};
    torch::nn::Sequential m{nullptr};
};

// SPPF Block (Spatial Pyramid Pooling - Fast)
class SPPF : public torch::nn::Module {
public:
    SPPF(int in_channels, int out_channels, int kernel_size = 5);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential cv1{nullptr};
    torch::nn::Sequential cv2{nullptr};
    torch::nn::MaxPool2d m{nullptr};
};

// YOLO Detection Head
class DetectionHead : public torch::nn::Module {
public:
    DetectionHead(int num_classes, const std::vector<int>& channels);
    std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& features);

private:
    torch::nn::ModuleList m{nullptr};
    int num_classes_;
    int num_outputs_;
};

// Main YOLO Model
class YoloModel : public torch::nn::Module {
public:
    explicit YoloModel(const YoloConfig& config);
    
    // Forward pass
    torch::Tensor forward(torch::Tensor x);
    
    // Forward with detection output
    std::vector<torch::Tensor> forward_detect(torch::Tensor x);
    
    // Post-process detections
    DetectionOutput post_process(const std::vector<torch::Tensor>& predictions);
    
    // Load pre-trained weights
    bool load_pretrained(const std::string& weights_path);
    
    // Save model
    bool save_model(const std::string& save_path);
    
    // Export to ONNX
    bool export_onnx(const std::string& onnx_path, int batch_size = 1);
    
    // Freeze backbone for fine-tuning
    void freeze_backbone();
    
    // Unfreeze all layers
    void unfreeze_all();
    
    const YoloConfig& config() const { return config_; }

private:
    void build_backbone();
    void build_neck();
    void build_head();
    
    YoloConfig config_;
    
    // Backbone (CSPDarknet)
    torch::nn::Sequential stem{nullptr};
    torch::nn::Sequential stage1{nullptr};
    torch::nn::Sequential stage2{nullptr};
    torch::nn::Sequential stage3{nullptr};
    torch::nn::Sequential stage4{nullptr};
    
    // Neck (PANet)
    torch::nn::Sequential neck{nullptr};
    
    // Head (Detection)
    std::shared_ptr<DetectionHead> head{nullptr};
};

// Non-Maximum Suppression
torch::Tensor nms(torch::Tensor boxes, torch::Tensor scores, float iou_threshold);

// YOLO loss computation
torch::Tensor compute_yolo_loss(
    const std::vector<torch::Tensor>& predictions,
    const torch::Tensor& targets,
    const YoloConfig& config
);

} // namespace yolo
