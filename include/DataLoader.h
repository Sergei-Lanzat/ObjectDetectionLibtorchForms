#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

namespace yolo {

// Annotation format
struct BoundingBox {
    float x;      // center x (normalized 0-1)
    float y;      // center y (normalized 0-1)
    float width;  // width (normalized 0-1)
    float height; // height (normalized 0-1)
    int class_id;
    float confidence = 1.0f;
};

// Image annotation
struct ImageAnnotation {
    std::string image_path;
    int image_width;
    int image_height;
    std::vector<BoundingBox> boxes;
};

// Dataset class
class YoloDataset : public torch::data::Dataset<YoloDataset> {
public:
    explicit YoloDataset(
        const std::string& data_path,
        int input_size = 640,
        bool augment = true
    );
    
    torch::data::Example<> get(size_t index) override;
    
    torch::optional<size_t> size() const override {
        return annotations_.size();
    }
    
    const std::vector<std::string>& get_class_names() const {
        return class_names_;
    }

private:
    torch::Tensor load_and_preprocess_image(const std::string& image_path);
    torch::Tensor augment_image(torch::Tensor image);
    void parse_annotation_file(const std::string& data_path);
    
    std::vector<ImageAnnotation> annotations_;
    std::vector<std::string> class_names_;
    int input_size_;
    bool augment_;
};

// Annotation format converters
class AnnotationConverter {
public:
    // Convert COCO format to YOLO format
    static bool coco_to_yolo(
        const std::string& coco_json_path,
        const std::string& output_dir
    );
    
    // Convert Pascal VOC format to YOLO format
    static bool voc_to_yolo(
        const std::string& voc_annotations_dir,
        const std::string& output_dir
    );
    
    // Convert LabelImg XML to YOLO format
    static bool labelimg_to_yolo(
        const std::string& xml_dir,
        const std::string& output_dir
    );
    
    // Validate YOLO format dataset
    static bool validate_yolo_dataset(const std::string& dataset_path);

private:
    static cv::Rect2f parse_voc_box(const std::string& xml_content);
    static void write_yolo_annotation(
        const std::string& output_path,
        const std::vector<BoundingBox>& boxes
    );
};

// Data augmentation utilities
class DataAugmentation {
public:
    static torch::Tensor random_flip(torch::Tensor image, std::vector<BoundingBox>& boxes);
    static torch::Tensor random_scale(torch::Tensor image, std::vector<BoundingBox>& boxes);
    static torch::Tensor random_crop(torch::Tensor image, std::vector<BoundingBox>& boxes);
    static torch::Tensor color_jitter(torch::Tensor image);
    static torch::Tensor mosaic_augmentation(
        const std::vector<torch::Tensor>& images,
        const std::vector<std::vector<BoundingBox>>& boxes
    );
};

} // namespace yolo
