#include "../include/YoloTrainer.h"
#include "../include/YoloModel.h"
#include "../include/DataLoader.h"
#include <iostream>
#include <filesystem>
#include <chrono>

namespace yolo {

YoloTrainer::YoloTrainer(
    std::shared_ptr<YoloModel> model,
    const TrainingConfig& config,
    torch::Device device
) : model_(model), config_(config), device_(device),
    should_stop_(false), is_training_(false),
    current_epoch_(0), total_iterations_(0) {
    
    setup_optimizer();
    setup_scheduler();
}

YoloTrainer::~YoloTrainer() {
    should_stop_ = true;
}

void YoloTrainer::setup_optimizer() {
    std::vector<torch::Tensor> parameters;
    for (auto& param : model_->parameters()) {
        if (param.requires_grad()) {
            parameters.push_back(param);
        }
    }
    
    if (config_.optimizer_type == "Adam") {
        optimizer_ = std::make_unique<torch::optim::Adam>(
            parameters,
            torch::optim::AdamOptions(config_.learning_rate)
                .weight_decay(config_.weight_decay)
        );
    } else if (config_.optimizer_type == "AdamW") {
        optimizer_ = std::make_unique<torch::optim::AdamW>(
            parameters,
            torch::optim::AdamWOptions(config_.learning_rate)
                .weight_decay(config_.weight_decay)
        );
    } else { // Default SGD
        optimizer_ = std::make_unique<torch::optim::SGD>(
            parameters,
            torch::optim::SGDOptions(config_.learning_rate)
                .momentum(config_.momentum)
                .weight_decay(config_.weight_decay)
        );
    }
}

void YoloTrainer::setup_scheduler() {
    // Placeholder for learning rate scheduler
    // In real implementation, would set up cosine annealing or step scheduler
}

bool YoloTrainer::train(
    const std::string& train_data_path,
    const std::string& val_data_path,
    ProgressCallback progress_callback
) {
    is_training_ = true;
    should_stop_ = false;
    
    try {
        // Create checkpoint directory
        std::filesystem::create_directories(config_.checkpoint_dir);
        
        // Load datasets
        std::cout << "Loading training dataset from: " << train_data_path << std::endl;
        auto train_dataset = YoloDataset(train_data_path, 
                                        model_->config().input_width, 
                                        true);
        
        auto train_loader = torch::data::make_data_loader(
            std::move(train_dataset),
            torch::data::DataLoaderOptions().batch_size(config_.batch_size).workers(config_.num_workers)
        );
        
        std::cout << "Starting training for " << config_.num_epochs << " epochs..." << std::endl;
        
        // Training loop
        for (int epoch = 0; epoch < config_.num_epochs && !should_stop_; epoch++) {
            current_epoch_ = epoch;
            model_->train();
            
            // Freeze backbone for initial epochs if configured
            if (config_.freeze_backbone && epoch < config_.freeze_epochs) {
                model_->freeze_backbone();
            } else if (config_.freeze_backbone && epoch == config_.freeze_epochs) {
                model_->unfreeze_all();
            }
            
            TrainingMetrics epoch_metrics;
            epoch_metrics.epoch = epoch;
            
            int batch_idx = 0;
            float running_loss = 0.0f;
            
            for (auto& batch : *train_loader) {
                if (should_stop_) break;
                
                optimizer_->zero_grad();
                
                auto data = batch.data.to(device_);
                auto target = batch.target.to(device_);
                
                // Forward pass
                auto predictions = model_->forward_detect(data);
                
                // Compute loss
                auto loss = compute_loss(predictions, target);
                
                // Backward pass
                loss.backward();
                optimizer_->step();
                
                // Update metrics
                running_loss += loss.item<float>();
                
                // Log progress
                if (batch_idx % config_.log_interval == 0) {
                    float avg_loss = running_loss / (batch_idx + 1);
                    epoch_metrics.loss = avg_loss;
                    epoch_metrics.iteration = total_iterations_;
                    
                    std::cout << "Epoch [" << epoch + 1 << "/" << config_.num_epochs << "] "
                             << "Batch [" << batch_idx << "] "
                             << "Loss: " << avg_loss << std::endl;
                    
                    if (progress_callback) {
                        progress_callback(epoch_metrics);
                    }
                }
                
                batch_idx++;
                total_iterations_++;
            }
            
            // Epoch completed
            epoch_metrics.loss = running_loss / batch_idx;
            history_.push_back(epoch_metrics);
            
            // Validation
            if (!val_data_path.empty()) {
                std::cout << "Running validation..." << std::endl;
                auto val_metrics = validate(val_data_path);
                std::cout << "Validation Loss: " << val_metrics.loss << std::endl;
            }
            
            // Save checkpoint
            if ((epoch + 1) % config_.save_interval == 0) {
                std::string checkpoint_path = config_.checkpoint_dir + "/checkpoint_epoch_" + 
                                            std::to_string(epoch + 1) + ".pt";
                save_checkpoint(checkpoint_path, epoch + 1);
            }
            
            if (progress_callback) {
                progress_callback(epoch_metrics);
            }
        }
        
        std::cout << "Training completed!" << std::endl;
        
        // Save final model
        std::string final_model_path = config_.checkpoint_dir + "/final_model.pt";
        model_->save_model(final_model_path);
        
        is_training_ = false;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Training error: " << e.what() << std::endl;
        is_training_ = false;
        return false;
    }
}

TrainingMetrics YoloTrainer::validate(const std::string& val_data_path) {
    model_->eval();
    TrainingMetrics metrics;
    
    try {
        auto val_dataset = YoloDataset(val_data_path, 
                                      model_->config().input_width, 
                                      false);
        
        auto val_loader = torch::data::make_data_loader(
            std::move(val_dataset),
            torch::data::DataLoaderOptions().batch_size(config_.batch_size)
        );
        
        torch::NoGradGuard no_grad;
        float total_loss = 0.0f;
        int num_batches = 0;
        
        for (auto& batch : *val_loader) {
            auto data = batch.data.to(device_);
            auto target = batch.target.to(device_);
            
            auto predictions = model_->forward_detect(data);
            auto loss = compute_loss(predictions, target);
            
            total_loss += loss.item<float>();
            num_batches++;
        }
        
        metrics.loss = total_loss / num_batches;
        
    } catch (const std::exception& e) {
        std::cerr << "Validation error: " << e.what() << std::endl;
    }
    
    model_->train();
    return metrics;
}

torch::Tensor YoloTrainer::compute_loss(
    const std::vector<torch::Tensor>& predictions,
    const torch::Tensor& targets
) {
    return compute_yolo_loss(predictions, targets, model_->config());
}

void YoloTrainer::update_metrics(const torch::Tensor& loss, TrainingMetrics& metrics) {
    metrics.loss = loss.item<float>();
    // Additional metrics computation would go here
}

bool YoloTrainer::save_checkpoint(const std::string& path, int epoch) {
    try {
        torch::serialize::OutputArchive archive;
        
        // Save model state
        model_->save(archive);
        
        // Save optimizer state
        archive.write("optimizer", optimizer_->state_dict());
        
        // Save epoch
        archive.write("epoch", torch::tensor(epoch));
        
        archive.save_to(path);
        
        std::cout << "Checkpoint saved: " << path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving checkpoint: " << e.what() << std::endl;
        return false;
    }
}

bool YoloTrainer::load_checkpoint(const std::string& path) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        
        // Load model state
        model_->load(archive);
        
        // Load optimizer state
        torch::IValue optimizer_state;
        archive.read("optimizer", optimizer_state);
        // optimizer_->load_state_dict(optimizer_state);
        
        // Load epoch
        torch::Tensor epoch_tensor;
        archive.read("epoch", epoch_tensor);
        current_epoch_ = epoch_tensor.item<int>();
        
        std::cout << "Checkpoint loaded: " << path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading checkpoint: " << e.what() << std::endl;
        return false;
    }
}

bool YoloTrainer::export_onnx(const std::string& onnx_path) {
    return model_->export_onnx(onnx_path);
}

} // namespace yolo
