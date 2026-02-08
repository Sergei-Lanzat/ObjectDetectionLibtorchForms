# Build Instructions

## Prerequisites

Before building this project, ensure you have the following installed:

### 1. Visual Studio
- **Version**: Visual Studio 2022 or later
- **Workloads Required**:
  - Desktop development with C++
  - .NET desktop development
  - C++/CLI support

### 2. LibTorch
1. Download LibTorch from https://pytorch.org/
   - Select: C++ / LibTorch
   - Choose CUDA version if you have NVIDIA GPU, or CPU-only
   
2. Extract the downloaded archive (e.g., `libtorch-win-shared-with-deps-2.0.0+cu118.zip`)
   - Recommended: extract directly to `ObjectDetectionLibtorchForms/libtorch` (CMake and the Visual Studio projects look here automatically)

3. Set environment variable:
   ```cmd
   setx Torch_DIR "C:\path\to\libtorch\share\cmake\Torch"
   ```

### 3. OpenCV
1. Download OpenCV from https://opencv.org/releases/
   - Version 4.x recommended

2. Extract and install

3. Set environment variable:
   ```cmd
   setx OpenCV_DIR "C:\path\to\opencv\build"
   ```

### 4. CMake
- Download from https://cmake.org/download/
- Version 3.18 or later
- Add to PATH during installation

### 5. nlohmann/json (JSON library)
This is header-only and can be installed via vcpkg or manually:
```cmd
vcpkg install nlohmann-json
```

## Build Steps

### Method 1: Using CMake (Recommended for Native Library)

1. **Open Command Prompt or PowerShell**

2. **Navigate to project directory**:
   ```cmd
   cd C:\path\to\ObjectDetectionLibtorchForms
   ```

3. **Create build directory**:
   ```cmd
   mkdir build
   cd build
   ```

4. **Configure with CMake**:
   ```cmd
   cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="C:\path\to\libtorch"
   ```
   
   If you have CUDA:
   ```cmd
   cmake .. -G "Visual Studio 17 2022" -A x64 ^
     -DCMAKE_PREFIX_PATH="C:\path\to\libtorch" ^
     -DCUDA_TOOLKIT_ROOT_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
   ```

5. **Build**:
   ```cmd
   cmake --build . --config Release
   ```

6. **Install (optional)**:
   ```cmd
   cmake --install . --prefix install
   ```

### Method 2: Using Visual Studio Solution

1. **Open Solution**:
   ```cmd
   start ObjectDetectionLibtorchForms.sln
   ```

2. **Configure Project Properties**:
   
   For **YoloTrainer** project:
   - Right-click → Properties
   - Configuration: All Configurations
   - Platform: x64
   
   **C/C++ → General → Additional Include Directories**:
   ```
   $(SolutionDir)include
   C:\path\to\libtorch\include
   C:\path\to\libtorch\include\torch\csrc\api\include
   C:\path\to\opencv\build\include
   ```
   
   **Linker → General → Additional Library Directories**:
   ```
   C:\path\to\libtorch\lib
   C:\path\to\opencv\build\x64\vc16\lib
   ```
   
   **Linker → Input → Additional Dependencies**:
   ```
   torch.lib
   torch_cpu.lib
   c10.lib
   opencv_world4xx.lib (replace xx with your version)
   ```

3. **Configure YoloTrainerGUI Project**:
   - Set as Startup Project
   - Configuration: Release, Platform: x64
   - Right-click → Properties
   
   **Common Language Runtime Support**:
   - Set to: Common Language Runtime Support (/clr)
   
   **Additional Dependencies**:
   - Add: `$(SolutionDir)\x64\Release\YoloTrainer.lib`

4. **Build Solution**:
   - Build → Build Solution (Ctrl+Shift+B)
   - Or: Build → Rebuild Solution

5. **Copy Required DLLs**:
   Copy these DLLs to the output directory (`bin\Release` or `x64\Release`):
   - From LibTorch: `torch.dll`, `c10.dll`, `torch_cpu.dll`, `torch_cuda.dll` (if CUDA)
   - From OpenCV: `opencv_world4xx.dll`

## Project Structure

```
ObjectDetectionLibtorchForms/
├── CMakeLists.txt              # CMake build configuration
├── ObjectDetectionLibtorchForms.sln  # Visual Studio solution
├── YoloTrainer/                # Native C++ library project
│   └── YoloTrainer.vcxproj
├── YoloTrainerGUI/             # C++/CLI WinForms GUI project
│   └── YoloTrainerGUI.vcxproj
├── include/                    # Header files
│   ├── YoloModel.h
│   ├── YoloTrainer.h
│   └── DataLoader.h
├── src/                        # Source files
│   ├── YoloTrainer/           # Native C++ implementation
│   │   ├── YoloModel.cpp
│   │   ├── YoloTrainer.cpp
│   │   └── DataLoader.cpp
│   └── GUI/                    # WinForms GUI
│       ├── MainForm.h
│       ├── MainForm.cpp
│       └── main.cpp
├── docs/                       # Documentation
│   ├── USER_GUIDE.md
│   └── BUILD_INSTRUCTIONS.md
├── examples/                   # Example datasets and configs
└── README.md
```

## Troubleshooting

### CMake Cannot Find LibTorch
- Ensure `CMAKE_PREFIX_PATH` points to LibTorch directory
- Check that `Torch_DIR` environment variable is set correctly
- Restart your terminal/IDE after setting environment variables

### Linker Errors (LNK2019, LNK2001)
- Verify all required libraries are listed in Additional Dependencies
- Check that library paths are correct
- Ensure Release/Debug configuration matches library configuration

### Runtime Error: Missing DLL
- Copy all required DLLs to the executable directory
- Add LibTorch and OpenCV bin directories to system PATH
- Use Dependency Walker to identify missing DLLs

### Cannot Open Include File
- Check Additional Include Directories in project properties
- Verify paths exist and are accessible
- Use absolute paths if relative paths don't work

### C++/CLI Compilation Errors
- Ensure Common Language Runtime Support is enabled
- Set Platform Toolset correctly (v143)
- Check .NET Framework target version (4.7.2 or later)

### CUDA Errors
- Install CUDA Toolkit matching your LibTorch version
- Update NVIDIA drivers
- Verify GPU is CUDA-capable: `nvidia-smi`

## Testing the Build

1. **Run the application**:
   ```cmd
   cd build\Release
   YoloTrainerGUI.exe
   ```

2. **Verify functionality**:
   - Application window opens
   - All UI elements are visible
   - No immediate crashes or errors

3. **Test with sample data**:
   - Use example dataset from `examples/` folder
   - Configure minimal settings
   - Start training to verify LibTorch integration

## Building for Distribution

1. **Build in Release mode**: Ensures optimized code
2. **Copy dependencies**: Include all DLLs
3. **Create installer** (optional): Use WiX or NSIS
4. **Test on clean system**: Verify no missing dependencies

## Performance Optimization

### For Faster Training
- Enable CUDA if available
- Use Release build (not Debug)
- Increase batch size if GPU memory allows
- Use mixed precision training (requires code modification)

### For Smaller Binary Size
- Use static linking (LibTorch static build)
- Remove unused dependencies
- Strip debug symbols in Release build

## Support

If you encounter build issues:
1. Check this document first
2. Search existing GitHub issues
3. Create new issue with:
   - Full error message
   - CMake/Build output
   - System information (OS, VS version, etc.)
   - Steps to reproduce
