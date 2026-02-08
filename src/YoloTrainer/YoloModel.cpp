#include "../include/YoloModel.h"
#include <torch/script.h>
#include <iostream>

namespace yolo {

// ConvBnSiLU implementation
ConvBnSiLU::ConvBnSiLU(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    conv = register_module("conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .bias(false)
    ));
    bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
    silu = register_module("silu", torch::nn::SiLU());
}

torch::Tensor ConvBnSiLU::forward(torch::Tensor x) {
    return silu->forward(bn->forward(conv->forward(x)));
}

// C3 Block implementation
C3::C3(int in_channels, int out_channels, int num_blocks, bool shortcut) {
    int hidden_channels = out_channels / 2;
    
    cv1 = register_module("cv1", torch::nn::Sequential(
        ConvBnSiLU(in_channels, hidden_channels, 1, 1, 0)
    ));
    
    cv2 = register_module("cv2", torch::nn::Sequential(
        ConvBnSiLU(in_channels, hidden_channels, 1, 1, 0)
    ));
    
    cv3 = register_module("cv3", torch::nn::Sequential(
        ConvBnSiLU(2 * hidden_channels, out_channels, 1, 1, 0)
    ));
    
    m = register_module("m", torch::nn::Sequential());
    for (int i = 0; i < num_blocks; i++) {
        m->push_back(torch::nn::Sequential(
            ConvBnSiLU(hidden_channels, hidden_channels, 3, 1, 1),
            ConvBnSiLU(hidden_channels, hidden_channels, 3, 1, 1)
        ));
    }
}

torch::Tensor C3::forward(torch::Tensor x) {
    auto y1 = cv1->forward(x);
    auto y2 = cv2->forward(x);
    
    for (const auto& module : *m) {
        y2 = module.forward(y2);
    }
    
    return cv3->forward(torch::cat({y1, y2}, 1));
}

// SPPF implementation
SPPF::SPPF(int in_channels, int out_channels, int kernel_size) {
    int hidden_channels = in_channels / 2;
    cv1 = register_module("cv1", torch::nn::Sequential(
        ConvBnSiLU(in_channels, hidden_channels, 1, 1, 0)
    ));
    cv2 = register_module("cv2", torch::nn::Sequential(
        ConvBnSiLU(hidden_channels * 4, out_channels, 1, 1, 0)
    ));
    m = register_module("m", torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(kernel_size).stride(1).padding(kernel_size / 2)
    ));
}

torch::Tensor SPPF::forward(torch::Tensor x) {
    x = cv1->forward(x);
    auto y1 = m->forward(x);
    auto y2 = m->forward(y1);
    auto y3 = m->forward(y2);
    return cv2->forward(torch::cat({x, y1, y2, y3}, 1));
}

// DetectionHead implementation
DetectionHead::DetectionHead(int num_classes, const std::vector<int>& channels) 
    : num_classes_(num_classes) {
    num_outputs_ = num_classes + 5; // 4 box coords + 1 objectness + num_classes
    m = register_module("m", torch::nn::ModuleList());
    
    for (size_t i = 0; i < channels.size(); i++) {
        m->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels[i], 3 * num_outputs_, 1)
        ));
    }
}

std::vector<torch::Tensor> DetectionHead::forward(const std::vector<torch::Tensor>& features) {
    std::vector<torch::Tensor> outputs;
    for (size_t i = 0; i < features.size(); i++) {
        auto out = m->at<torch::nn::Conv2d>(i)->forward(features[i]);
        outputs.push_back(out);
    }
    return outputs;
}

// YoloModel implementation
YoloModel::YoloModel(const YoloConfig& config) : config_(config) {
    build_backbone();
    build_neck();
    build_head();
}

void YoloModel::build_backbone() {
    // Stem
    stem = register_module("stem", torch::nn::Sequential(
        ConvBnSiLU(3, 32, 6, 2, 2),  // P1/2
        ConvBnSiLU(32, 64, 3, 2, 1)  // P2/4
    ));
    
    // Stage 1
    stage1 = register_module("stage1", torch::nn::Sequential(
        C3(64, 64, 1),
        ConvBnSiLU(64, 128, 3, 2, 1)  // P3/8
    ));
    
    // Stage 2
    stage2 = register_module("stage2", torch::nn::Sequential(
        C3(128, 128, 2),
        ConvBnSiLU(128, 256, 3, 2, 1)  // P4/16
    ));
    
    // Stage 3
    stage3 = register_module("stage3", torch::nn::Sequential(
        C3(256, 256, 3),
        ConvBnSiLU(256, 512, 3, 2, 1)  // P5/32
    ));
    
    // Stage 4
    stage4 = register_module("stage4", torch::nn::Sequential(
        C3(512, 512, 1),
        SPPF(512, 512, 5)
    ));
}

void YoloModel::build_neck() {
    // Simplified PANet neck
    neck = register_module("neck", torch::nn::Sequential(
        ConvBnSiLU(512, 256, 1, 1, 0),
        torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{2, 2}).mode(torch::kNearest))
    ));
}

void YoloModel::build_head() {
    std::vector<int> channels = {256, 256, 512}; // Channels for different detection scales
    head = std::make_shared<DetectionHead>(config_.num_classes, channels);
    register_module("head", head);
}

torch::Tensor YoloModel::forward(torch::Tensor x) {
    // Backbone
    auto p1 = stem->forward(x);
    auto p2 = stage1->forward(p1);
    auto p3 = stage2->forward(p2);
    auto p4 = stage3->forward(p3);
    auto p5 = stage4->forward(p4);
    
    // For training, return raw feature maps
    // For inference, use forward_detect
    return p5;
}

std::vector<torch::Tensor> YoloModel::forward_detect(torch::Tensor x) {
    // Get backbone features
    auto p1 = stem->forward(x);
    auto p2 = stage1->forward(p1);
    auto p3 = stage2->forward(p2);
    auto p4 = stage3->forward(p3);
    auto p5 = stage4->forward(p4);
    
    // Create feature pyramid
    std::vector<torch::Tensor> features = {p3, p4, p5};
    
    // Detection head
    return head->forward(features);
}

DetectionOutput YoloModel::post_process(const std::vector<torch::Tensor>& predictions) {
    DetectionOutput output;
    
    // Simplified post-processing
    // In a real implementation, this would decode the predictions,
    // apply NMS, and filter by confidence
    
    std::vector<torch::Tensor> all_boxes;
    std::vector<torch::Tensor> all_scores;
    std::vector<torch::Tensor> all_class_ids;
    
    for (const auto& pred : predictions) {
        // Decode predictions (simplified)
        auto batch_size = pred.size(0);
        auto num_anchors = pred.size(2);
        auto grid_h = pred.size(3);
        auto grid_w = pred.size(4);
        
        // Extract boxes, objectness, and class scores
        // This is a placeholder - real implementation would be more complex
    }
    
    // Concatenate and apply NMS
    if (!all_boxes.empty()) {
        output.boxes = torch::cat(all_boxes, 0);
        output.scores = torch::cat(all_scores, 0);
        output.class_ids = torch::cat(all_class_ids, 0);
        
        // Apply NMS
        auto keep_indices = nms(output.boxes, output.scores, config_.iou_threshold);
        output.boxes = output.boxes.index_select(0, keep_indices);
        output.scores = output.scores.index_select(0, keep_indices);
        output.class_ids = output.class_ids.index_select(0, keep_indices);
    }
    
    return output;
}

bool YoloModel::load_pretrained(const std::string& weights_path) {
    try {
        torch::load(this, weights_path);
        std::cout << "Loaded pre-trained weights from: " << weights_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        return false;
    }
}

bool YoloModel::save_model(const std::string& save_path) {
    try {
        torch::save(this, save_path);
        std::cout << "Model saved to: " << save_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return false;
    }
}

bool YoloModel::export_onnx(const std::string& onnx_path, int batch_size) {
    try {
        // Set model to eval mode
        this->eval();
        
        // Create dummy input
        auto dummy_input = torch::randn({batch_size, 3, config_.input_height, config_.input_width});
        
        // Trace the model
        auto traced_model = torch::jit::trace(this, dummy_input);
        
        // Save as TorchScript (ONNX export requires torch::jit::script or external tools)
        traced_model.save(onnx_path + ".pt");
        
        std::cout << "Model exported to: " << onnx_path << ".pt" << std::endl;
        std::cout << "Note: For ONNX export, use Python torch.onnx.export or TorchScript" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error exporting model: " << e.what() << std::endl;
        return false;
    }
}

void YoloModel::freeze_backbone() {
    // Freeze stem and stages
    for (auto& param : stem->parameters()) {
        param.set_requires_grad(false);
    }
    for (auto& param : stage1->parameters()) {
        param.set_requires_grad(false);
    }
    for (auto& param : stage2->parameters()) {
        param.set_requires_grad(false);
    }
    for (auto& param : stage3->parameters()) {
        param.set_requires_grad(false);
    }
    for (auto& param : stage4->parameters()) {
        param.set_requires_grad(false);
    }
    std::cout << "Backbone frozen for fine-tuning" << std::endl;
}

void YoloModel::unfreeze_all() {
    for (auto& param : this->parameters()) {
        param.set_requires_grad(true);
    }
    std::cout << "All layers unfrozen" << std::endl;
}

// NMS implementation
torch::Tensor nms(torch::Tensor boxes, torch::Tensor scores, float iou_threshold) {
    if (boxes.size(0) == 0) {
        return torch::empty({0}, torch::kLong);
    }
    
    // Use torchvision NMS if available, otherwise implement custom
    auto x1 = boxes.select(1, 0);
    auto y1 = boxes.select(1, 1);
    auto x2 = boxes.select(1, 2);
    auto y2 = boxes.select(1, 3);
    
    auto areas = (x2 - x1) * (y2 - y1);
    auto order = std::get<1>(scores.sort(0, true));
    
    std::vector<int64_t> keep;
    
    while (order.numel() > 0) {
        if (order.numel() == 1) {
            keep.push_back(order.item<int64_t>());
            break;
        }
        
        auto i = order[0].item<int64_t>();
        keep.push_back(i);
        
        // Compute IoU
        auto xx1 = torch::max(x1[i], x1.index_select(0, order.slice(0, 1)));
        auto yy1 = torch::max(y1[i], y1.index_select(0, order.slice(0, 1)));
        auto xx2 = torch::min(x2[i], x2.index_select(0, order.slice(0, 1)));
        auto yy2 = torch::min(y2[i], y2.index_select(0, order.slice(0, 1)));
        
        auto w = torch::clamp(xx2 - xx1, 0);
        auto h = torch::clamp(yy2 - yy1, 0);
        auto inter = w * h;
        
        auto iou = inter / (areas[i] + areas.index_select(0, order.slice(0, 1)) - inter);
        
        auto mask = iou <= iou_threshold;
        auto indices = torch::nonzero(mask).squeeze(1);
        
        if (indices.numel() == 0) {
            break;
        }
        
        order = order.index_select(0, indices + 1);
    }
    
    return torch::tensor(keep, torch::kLong);
}

// YOLO loss computation
torch::Tensor compute_yolo_loss(
    const std::vector<torch::Tensor>& predictions,
    const torch::Tensor& targets,
    const YoloConfig& config
) {
    // Simplified loss computation
    // Real implementation would compute box loss, objectness loss, and classification loss
    
    torch::Tensor total_loss = torch::zeros({1});
    
    for (const auto& pred : predictions) {
        // Compute losses for each scale
        // This is a placeholder
        total_loss = total_loss + pred.mean();
    }
    
    return total_loss;
}

} // namespace yolo
