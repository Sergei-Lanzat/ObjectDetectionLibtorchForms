#include "../include/DataLoader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <nlohmann/json.hpp>

namespace yolo {

// YoloDataset implementation
YoloDataset::YoloDataset(
    const std::string& data_path,
    int input_size,
    bool augment
) : input_size_(input_size), augment_(augment) {
    parse_annotation_file(data_path);
}

torch::data::Example<> YoloDataset::get(size_t index) {
    const auto& annotation = annotations_[index];
    
    // Load image
    auto image = load_and_preprocess_image(annotation.image_path);
    
    // Apply augmentation if enabled
    if (augment_) {
        image = augment_image(image);
    }
    
    // Convert boxes to tensor
    std::vector<float> box_data;
    for (const auto& box : annotation.boxes) {
        box_data.push_back(box.class_id);
        box_data.push_back(box.x);
        box_data.push_back(box.y);
        box_data.push_back(box.width);
        box_data.push_back(box.height);
    }
    
    torch::Tensor target;
    if (!box_data.empty()) {
        target = torch::from_blob(box_data.data(), 
                                 {static_cast<long>(annotation.boxes.size()), 5},
                                 torch::kFloat32).clone();
    } else {
        target = torch::zeros({0, 5});
    }
    
    return {image, target};
}

torch::Tensor YoloDataset::load_and_preprocess_image(const std::string& image_path) {
    // Load image using OpenCV
    cv::Mat img = cv::imread(image_path);
    
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    
    // Resize to input size
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size_, input_size_));
    
    // Convert BGR to RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    
    // Convert to float and normalize to [0, 1]
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    
    // Convert to tensor [C, H, W]
    torch::Tensor tensor = torch::from_blob(
        resized.data,
        {resized.rows, resized.cols, 3},
        torch::kFloat32
    ).clone();
    
    // Permute to [C, H, W]
    tensor = tensor.permute({2, 0, 1});
    
    return tensor;
}

torch::Tensor YoloDataset::augment_image(torch::Tensor image) {
    // Simple augmentation: random flip
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    
    if (dis(gen) > 0.5) {
        image = torch::flip(image, {2}); // Horizontal flip
    }
    
    return image;
}

void YoloDataset::parse_annotation_file(const std::string& data_path) {
    // Parse YOLO format dataset
    // Expected structure:
    // data_path/
    //   images/
    //   labels/
    //   classes.txt
    
    std::string images_dir = data_path + "/images";
    std::string labels_dir = data_path + "/labels";
    std::string classes_file = data_path + "/classes.txt";
    
    // Load class names
    std::ifstream classes_stream(classes_file);
    if (classes_stream.is_open()) {
        std::string line;
        while (std::getline(classes_stream, line)) {
            if (!line.empty()) {
                class_names_.push_back(line);
            }
        }
    }
    
    // Scan for images
    for (const auto& entry : std::filesystem::directory_iterator(images_dir)) {
        if (entry.is_regular_file()) {
            std::string img_path = entry.path().string();
            std::string img_name = entry.path().stem().string();
            
            // Look for corresponding label file
            std::string label_path = labels_dir + "/" + img_name + ".txt";
            
            ImageAnnotation annotation;
            annotation.image_path = img_path;
            
            // Load image to get dimensions
            cv::Mat img = cv::imread(img_path);
            if (!img.empty()) {
                annotation.image_width = img.cols;
                annotation.image_height = img.rows;
                
                // Parse label file
                std::ifstream label_stream(label_path);
                if (label_stream.is_open()) {
                    std::string line;
                    while (std::getline(label_stream, line)) {
                        std::istringstream iss(line);
                        BoundingBox box;
                        if (iss >> box.class_id >> box.x >> box.y >> box.width >> box.height) {
                            annotation.boxes.push_back(box);
                        }
                    }
                }
                
                annotations_.push_back(annotation);
            }
        }
    }
    
    std::cout << "Loaded " << annotations_.size() << " images" << std::endl;
}

// AnnotationConverter implementation
bool AnnotationConverter::coco_to_yolo(
    const std::string& coco_json_path,
    const std::string& output_dir
) {
    try {
        std::ifstream file(coco_json_path);
        nlohmann::json coco_data;
        file >> coco_data;
        
        std::filesystem::create_directories(output_dir + "/labels");
        
        // Create class names file
        std::ofstream classes_file(output_dir + "/classes.txt");
        for (const auto& category : coco_data["categories"]) {
            classes_file << category["name"].get<std::string>() << "\n";
        }
        classes_file.close();
        
        // Convert annotations
        for (const auto& annotation : coco_data["annotations"]) {
            int image_id = annotation["image_id"];
            int category_id = annotation["category_id"];
            
            auto bbox = annotation["bbox"];
            float x = bbox[0].get<float>();
            float y = bbox[1].get<float>();
            float w = bbox[2].get<float>();
            float h = bbox[3].get<float>();
            
            // Find image info to get image dimensions
            int img_width = 0, img_height = 0;
            for (const auto& image : coco_data["images"]) {
                if (image["id"] == image_id) {
                    img_width = image["width"];
                    img_height = image["height"];
                    break;
                }
            }
            
            if (img_width > 0 && img_height > 0) {
                // Convert to YOLO format (normalized center coordinates)
                float x_center = (x + w / 2.0f) / img_width;
                float y_center = (y + h / 2.0f) / img_height;
                float norm_w = w / img_width;
                float norm_h = h / img_height;
                
                // Write to label file
                std::string label_file = output_dir + "/labels/" + 
                                        std::to_string(image_id) + ".txt";
                std::ofstream label_stream(label_file, std::ios::app);
                label_stream << category_id << " " 
                           << x_center << " " << y_center << " "
                           << norm_w << " " << norm_h << "\n";
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error converting COCO to YOLO: " << e.what() << std::endl;
        return false;
    }
}

bool AnnotationConverter::voc_to_yolo(
    const std::string& voc_annotations_dir,
    const std::string& output_dir
) {
    // Implementation for VOC to YOLO conversion
    // This would parse XML files and convert to YOLO format
    std::filesystem::create_directories(output_dir + "/labels");
    
    std::cout << "VOC to YOLO conversion not fully implemented" << std::endl;
    return false;
}

bool AnnotationConverter::labelimg_to_yolo(
    const std::string& xml_dir,
    const std::string& output_dir
) {
    // Similar to VOC format
    return voc_to_yolo(xml_dir, output_dir);
}

bool AnnotationConverter::validate_yolo_dataset(const std::string& dataset_path) {
    // Check if dataset has required structure
    std::string images_dir = dataset_path + "/images";
    std::string labels_dir = dataset_path + "/labels";
    std::string classes_file = dataset_path + "/classes.txt";
    
    bool valid = std::filesystem::exists(images_dir) &&
                 std::filesystem::exists(labels_dir) &&
                 std::filesystem::exists(classes_file);
    
    if (valid) {
        std::cout << "Dataset structure is valid" << std::endl;
    } else {
        std::cerr << "Invalid dataset structure" << std::endl;
    }
    
    return valid;
}

// DataAugmentation implementation
torch::Tensor DataAugmentation::random_flip(torch::Tensor image, std::vector<BoundingBox>& boxes) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    
    if (dis(gen) > 0.5) {
        image = torch::flip(image, {2}); // Horizontal flip
        
        // Flip bounding boxes
        for (auto& box : boxes) {
            box.x = 1.0f - box.x;
        }
    }
    
    return image;
}

torch::Tensor DataAugmentation::random_scale(torch::Tensor image, std::vector<BoundingBox>& boxes) {
    // Scale augmentation implementation
    return image;
}

torch::Tensor DataAugmentation::random_crop(torch::Tensor image, std::vector<BoundingBox>& boxes) {
    // Crop augmentation implementation
    return image;
}

torch::Tensor DataAugmentation::color_jitter(torch::Tensor image) {
    // Color jittering implementation
    return image;
}

torch::Tensor DataAugmentation::mosaic_augmentation(
    const std::vector<torch::Tensor>& images,
    const std::vector<std::vector<BoundingBox>>& boxes
) {
    // Mosaic augmentation (combine 4 images into one)
    // This is a placeholder
    return images[0];
}

} // namespace yolo
