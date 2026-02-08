#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace yolo {

class YoloModel;
struct YoloConfig;

// Training configuration
struct TrainingConfig {
    int num_epochs = 100;
    int batch_size = 16;
    float learning_rate = 0.01f;
    float weight_decay = 0.0005f;
    float momentum = 0.937f;
    int warmup_epochs = 3;
    bool use_mosaic = true;
    bool use_mixup = false;
    std::string optimizer_type = "SGD"; // SGD, Adam, AdamW
    std::string scheduler_type = "cosine"; // cosine, step, none
    int save_interval = 10;
    int log_interval = 10;
    std::string checkpoint_dir = "checkpoints";
    bool resume_training = false;
    std::string resume_checkpoint = "";
    int num_workers = 4;
    bool freeze_backbone = false;
    int freeze_epochs = 0;
};

// Training metrics
struct TrainingMetrics {
    float loss = 0.0f;
    float box_loss = 0.0f;
    float obj_loss = 0.0f;
    float cls_loss = 0.0f;
    float precision = 0.0f;
    float recall = 0.0f;
    float map50 = 0.0f;
    float map = 0.0f;
    int epoch = 0;
    int iteration = 0;
};

// Callback function type for progress updates
using ProgressCallback = std::function<void(const TrainingMetrics&)>;

// YOLO Trainer class
class YoloTrainer {
public:
    YoloTrainer(
        std::shared_ptr<YoloModel> model,
        const TrainingConfig& config,
        torch::Device device = torch::kCPU
    );
    
    ~YoloTrainer();
    
    // Start training
    bool train(
        const std::string& train_data_path,
        const std::string& val_data_path,
        ProgressCallback progress_callback = nullptr
    );
    
    // Validate model
    TrainingMetrics validate(const std::string& val_data_path);
    
    // Save checkpoint
    bool save_checkpoint(const std::string& path, int epoch);
    
    // Load checkpoint
    bool load_checkpoint(const std::string& path);
    
    // Export trained model to ONNX
    bool export_onnx(const std::string& onnx_path);
    
    // Get training history
    const std::vector<TrainingMetrics>& get_history() const { return history_; }
    
    // Stop training
    void stop_training() { should_stop_ = true; }
    
    // Check if training is in progress
    bool is_training() const { return is_training_; }

private:
    void setup_optimizer();
    void setup_scheduler();
    torch::Tensor compute_loss(
        const std::vector<torch::Tensor>& predictions,
        const torch::Tensor& targets
    );
    void update_metrics(const torch::Tensor& loss, TrainingMetrics& metrics);
    
    std::shared_ptr<YoloModel> model_;
    TrainingConfig config_;
    torch::Device device_;
    
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    std::unique_ptr<torch::optim::LRScheduler> scheduler_;
    
    std::vector<TrainingMetrics> history_;
    bool should_stop_;
    bool is_training_;
    
    int current_epoch_;
    int total_iterations_;
};

} // namespace yolo
