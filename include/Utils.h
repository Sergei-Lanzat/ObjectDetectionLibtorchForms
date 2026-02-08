#pragma once

#include <string>
#include <vector>
#include <memory>

namespace yolo {
namespace utils {

// File system utilities
class FileSystem {
public:
    static bool exists(const std::string& path);
    static bool is_directory(const std::string& path);
    static bool is_file(const std::string& path);
    static bool create_directory(const std::string& path);
    static bool create_directories(const std::string& path);
    static std::vector<std::string> list_files(const std::string& directory, const std::string& extension = "");
    static std::string get_filename(const std::string& path);
    static std::string get_extension(const std::string& path);
    static std::string get_parent_directory(const std::string& path);
    static std::string join_path(const std::string& path1, const std::string& path2);
};

// String utilities
class StringUtils {
public:
    static std::string to_lower(const std::string& str);
    static std::string to_upper(const std::string& str);
    static std::vector<std::string> split(const std::string& str, char delimiter);
    static std::string trim(const std::string& str);
    static bool starts_with(const std::string& str, const std::string& prefix);
    static bool ends_with(const std::string& str, const std::string& suffix);
    static std::string replace(const std::string& str, const std::string& from, const std::string& to);
};

// Image utilities
class ImageUtils {
public:
    // Convert BGR to RGB
    static void bgr_to_rgb(unsigned char* data, int width, int height);
    
    // Resize image
    static void resize_image(unsigned char* src, int src_w, int src_h,
                           unsigned char* dst, int dst_w, int dst_h, int channels);
    
    // Normalize image to [0, 1]
    static void normalize(unsigned char* data, float* output, int width, int height, int channels);
    
    // Draw bounding box
    static void draw_box(unsigned char* image, int img_w, int img_h,
                        int x, int y, int w, int h,
                        unsigned char r, unsigned char g, unsigned char b);
};

// Logging utilities
class Logger {
public:
    enum class Level {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        FATAL
    };
    
    static void set_level(Level level);
    static void log(Level level, const std::string& message);
    static void debug(const std::string& message);
    static void info(const std::string& message);
    static void warning(const std::string& message);
    static void error(const std::string& message);
    static void fatal(const std::string& message);
    
private:
    static Level current_level_;
};

// Timer for performance measurement
class Timer {
public:
    Timer();
    void start();
    void stop();
    double elapsed_seconds() const;
    double elapsed_milliseconds() const;
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_;
    bool is_running_;
};

// Progress bar for console output
class ProgressBar {
public:
    explicit ProgressBar(int total, int width = 50);
    void update(int current);
    void finish();
    
private:
    int total_;
    int width_;
    int current_;
};

} // namespace utils
} // namespace yolo
