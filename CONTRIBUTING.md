# Contributing to ObjectDetectionLibtorchForms

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

When reporting bugs, please include:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- System information (OS, Visual Studio version, LibTorch version)
- Error messages and logs
- Screenshots if applicable

### Suggesting Features

Feature suggestions are welcome! Please:
- Check if the feature has already been requested
- Provide a clear description of the feature
- Explain the use case and benefits
- Consider implementation complexity

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**:
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Test your changes**:
   - Build the project successfully
   - Test the functionality
   - Verify no regressions

5. **Commit your changes**:
   ```bash
   git commit -m "Add amazing feature"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**

## Code Style Guidelines

### C++ Code Style
- Use C++17 features
- Follow Google C++ Style Guide (mostly)
- Use meaningful variable names
- Keep functions focused and short
- Add comments for non-obvious code

### Example:
```cpp
// Good
float calculate_iou(const BoundingBox& box1, const BoundingBox& box2) {
    float intersection = compute_intersection(box1, box2);
    float union_area = box1.area() + box2.area() - intersection;
    return intersection / union_area;
}

// Avoid
float calc(const BoundingBox& b1, const BoundingBox& b2) {
    float i = compute_intersection(b1, b2);
    float u = b1.area() + b2.area() - i;
    return i / u;
}
```

### C++/CLI GUI Code
- Separate UI logic from business logic
- Use meaningful event handler names
- Keep UI responsive (use background threads)
- Handle errors gracefully with user-friendly messages

### Documentation
- Update README.md for major features
- Add comments to public APIs
- Update USER_GUIDE.md for user-facing changes
- Include code examples where helpful

## Development Setup

1. **Install prerequisites**:
   - Visual Studio 2022+
   - LibTorch
   - OpenCV
   - CMake

2. **Clone and build**:
   ```bash
   git clone https://github.com/Sergei-Lanzat/ObjectDetectionLibtorchForms.git
   cd ObjectDetectionLibtorchForms
   mkdir build && cd build
   cmake ..
   cmake --build .
   ```

3. **Run tests** (if available):
   ```bash
   ctest
   ```

## Areas for Contribution

### High Priority
- [ ] Unit tests for core components
- [ ] Integration tests for training pipeline
- [ ] Performance optimizations
- [ ] Memory leak fixes
- [ ] Cross-platform support (Linux, macOS)

### Medium Priority
- [ ] Additional YOLO versions (YOLOv9, YOLOv10)
- [ ] More data augmentation techniques
- [ ] TensorBoard integration
- [ ] Model quantization
- [ ] Multi-GPU support

### Low Priority (Nice to Have)
- [ ] Dark mode for GUI
- [ ] Localization (multiple languages)
- [ ] Plugin system for custom models
- [ ] Cloud training integration
- [ ] Model zoo with pre-trained models

## Code Review Process

1. **Automated checks**:
   - Build must succeed
   - No compiler warnings
   - Code formatting checks

2. **Manual review**:
   - Code quality and style
   - Documentation completeness
   - Test coverage
   - Performance considerations

3. **Approval**:
   - At least one maintainer approval required
   - Address all review comments
   - Rebase on main if needed

## Testing Guidelines

### Unit Tests
- Test individual functions
- Cover edge cases
- Use mock objects where appropriate

### Integration Tests
- Test component interactions
- Verify data flow
- Test error handling

### Manual Testing
- Test GUI functionality
- Verify training works end-to-end
- Check ONNX export
- Test on different datasets

## Documentation Standards

### Code Comments
```cpp
/**
 * @brief Computes Intersection over Union (IoU) between two bounding boxes.
 * 
 * @param box1 First bounding box
 * @param box2 Second bounding box
 * @return IoU score in range [0, 1]
 */
float calculate_iou(const BoundingBox& box1, const BoundingBox& box2);
```

### Markdown Documentation
- Use clear headings
- Include code examples
- Add screenshots for UI features
- Link to related documentation

## Questions?

- Open a GitHub issue with the "question" label
- Check existing issues and documentation first
- Be respectful and patient

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉
