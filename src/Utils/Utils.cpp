#include "../include/Utils.h"
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>

namespace yolo {
namespace utils {

// FileSystem implementation
bool FileSystem::exists(const std::string& path) {
    return std::filesystem::exists(path);
}

bool FileSystem::is_directory(const std::string& path) {
    return std::filesystem::is_directory(path);
}

bool FileSystem::is_file(const std::string& path) {
    return std::filesystem::is_regular_file(path);
}

bool FileSystem::create_directory(const std::string& path) {
    return std::filesystem::create_directory(path);
}

bool FileSystem::create_directories(const std::string& path) {
    return std::filesystem::create_directories(path);
}

std::vector<std::string> FileSystem::list_files(const std::string& directory, const std::string& extension) {
    std::vector<std::string> files;
    
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string filepath = entry.path().string();
            if (extension.empty() || ends_with(filepath, extension)) {
                files.push_back(filepath);
            }
        }
    }
    
    return files;
}

std::string FileSystem::get_filename(const std::string& path) {
    return std::filesystem::path(path).filename().string();
}

std::string FileSystem::get_extension(const std::string& path) {
    return std::filesystem::path(path).extension().string();
}

std::string FileSystem::get_parent_directory(const std::string& path) {
    return std::filesystem::path(path).parent_path().string();
}

std::string FileSystem::join_path(const std::string& path1, const std::string& path2) {
    return (std::filesystem::path(path1) / path2).string();
}

// StringUtils implementation
std::string StringUtils::to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string StringUtils::to_upper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

std::vector<std::string> StringUtils::split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::string StringUtils::trim(const std::string& str) {
    const char* whitespace = " \t\n\r\f\v";
    size_t start = str.find_first_not_of(whitespace);
    size_t end = str.find_last_not_of(whitespace);
    
    if (start == std::string::npos) {
        return "";
    }
    
    return str.substr(start, end - start + 1);
}

bool StringUtils::starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && 
           str.compare(0, prefix.size(), prefix) == 0;
}

bool StringUtils::ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string StringUtils::replace(const std::string& str, const std::string& from, const std::string& to) {
    std::string result = str;
    size_t pos = 0;
    
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    
    return result;
}

// ImageUtils implementation
void ImageUtils::bgr_to_rgb(unsigned char* data, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        std::swap(data[i * 3], data[i * 3 + 2]);
    }
}

void ImageUtils::resize_image(unsigned char* src, int src_w, int src_h,
                             unsigned char* dst, int dst_w, int dst_h, int channels) {
    // Simple nearest neighbor resize
    float x_ratio = static_cast<float>(src_w) / dst_w;
    float y_ratio = static_cast<float>(src_h) / dst_h;
    
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            int src_x = static_cast<int>(x * x_ratio);
            int src_y = static_cast<int>(y * y_ratio);
            
            for (int c = 0; c < channels; c++) {
                dst[(y * dst_w + x) * channels + c] = 
                    src[(src_y * src_w + src_x) * channels + c];
            }
        }
    }
}

void ImageUtils::normalize(unsigned char* data, float* output, int width, int height, int channels) {
    int total = width * height * channels;
    for (int i = 0; i < total; i++) {
        output[i] = static_cast<float>(data[i]) / 255.0f;
    }
}

void ImageUtils::draw_box(unsigned char* image, int img_w, int img_h,
                         int x, int y, int w, int h,
                         unsigned char r, unsigned char g, unsigned char b) {
    // Draw rectangle (simplified)
    for (int i = 0; i < w; i++) {
        if (x + i < img_w && y >= 0 && y < img_h) {
            image[(y * img_w + x + i) * 3 + 0] = r;
            image[(y * img_w + x + i) * 3 + 1] = g;
            image[(y * img_w + x + i) * 3 + 2] = b;
            
            if (y + h < img_h) {
                image[((y + h) * img_w + x + i) * 3 + 0] = r;
                image[((y + h) * img_w + x + i) * 3 + 1] = g;
                image[((y + h) * img_w + x + i) * 3 + 2] = b;
            }
        }
    }
    
    for (int i = 0; i < h; i++) {
        if (x >= 0 && x < img_w && y + i < img_h) {
            image[((y + i) * img_w + x) * 3 + 0] = r;
            image[((y + i) * img_w + x) * 3 + 1] = g;
            image[((y + i) * img_w + x) * 3 + 2] = b;
            
            if (x + w < img_w) {
                image[((y + i) * img_w + x + w) * 3 + 0] = r;
                image[((y + i) * img_w + x + w) * 3 + 1] = g;
                image[((y + i) * img_w + x + w) * 3 + 2] = b;
            }
        }
    }
}

// Logger implementation
Logger::Level Logger::current_level_ = Logger::Level::INFO;

void Logger::set_level(Level level) {
    current_level_ = level;
}

void Logger::log(Level level, const std::string& message) {
    if (level < current_level_) return;
    
    const char* level_str[] = {"DEBUG", "INFO", "WARNING", "ERROR", "FATAL"};
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    
    std::cout << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "] "
              << "[" << level_str[static_cast<int>(level)] << "] "
              << message << std::endl;
}

void Logger::debug(const std::string& message) {
    log(Level::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(Level::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(Level::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(Level::ERROR, message);
}

void Logger::fatal(const std::string& message) {
    log(Level::FATAL, message);
}

// Timer implementation
Timer::Timer() : is_running_(false) {}

void Timer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    is_running_ = true;
}

void Timer::stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
    is_running_ = false;
}

double Timer::elapsed_seconds() const {
    auto end = is_running_ ? std::chrono::high_resolution_clock::now() : end_time_;
    return std::chrono::duration<double>(end - start_time_).count();
}

double Timer::elapsed_milliseconds() const {
    return elapsed_seconds() * 1000.0;
}

// ProgressBar implementation
ProgressBar::ProgressBar(int total, int width) 
    : total_(total), width_(width), current_(0) {}

void ProgressBar::update(int current) {
    current_ = current;
    float progress = static_cast<float>(current) / total_;
    int pos = static_cast<int>(width_ * progress);
    
    std::cout << "[";
    for (int i = 0; i < width_; i++) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "%\r";
    std::cout.flush();
}

void ProgressBar::finish() {
    update(total_);
    std::cout << std::endl;
}

} // namespace utils
} // namespace yolo
