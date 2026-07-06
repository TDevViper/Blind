# Building & Running the Native C++ Port (`BlindAssistant_VC.cpp`)

The native C++ port (`BlindAssistant_VC.cpp`) is a standalone, ultra-low latency implementation of the BlindAssistive AI™ tracking pipeline. It uses the OpenCV C++ API for video capture and DNN inference, combined with Windows Native Speech API (SAPI) via COM interfaces for asynchronous speech guidance.

## Prerequisites
1. **Windows 10 / 11 (x64)**
2. **Visual Studio 2019 / 2022** with "Desktop development with C++" workload installed.
3. **CMake** (v3.15 or newer, included with Visual Studio or downloadable from [cmake.org](https://cmake.org/)).
4. **OpenCV 4.x for Windows** (pre-built Win pack from [opencv.org](https://opencv.org/releases/)):
   - Extract OpenCV to a directory on your system (e.g., `C:\opencv`).
   - Note the build path, e.g., `C:\opencv\build\x64\vc16` (or `vc17`).

## Build Instructions using CMake (Recommended)

1. **Open developer Command Prompt or PowerShell** and navigate to the root `BLIND` project directory:
   ```powershell
   cd path\to\BLIND
   ```

2. **Create a build directory**:
   ```powershell
   mkdir build
   cd build
   ```

3. **Configure the project with CMake**, specifying your `OpenCV_DIR`:
   ```powershell
   cmake .. -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:\opencv\build\x64\vc16"
   ```
   *(Note: Replace `Visual Studio 17 2022` with `Visual Studio 16 2019` if using VS 2019, and adjust the `OpenCV_DIR` path to match your installation).*

4. **Build the executable in Release mode**:
   ```powershell
   cmake --build . --config Release
   ```

5. **Run the Application**:
   Ensure `opencv_world4xx.dll` is in your system PATH or copied next to the generated `.exe`, then run:
   ```powershell
   .\Release\BlindAssistantVC.exe
   ```

## Build Instructions via Visual Studio GUI (Direct Solution Project)
1. In Visual Studio, select **File -> Open -> Folder...** and choose the root `BLIND` folder.
2. Visual Studio will detect `CMakeLists.txt` automatically.
3. In **Project -> CMake Settings**, add an environment variable or path setting for `OpenCV_DIR`.
4. Select the **x64-Release** build target and click **Build -> Build All**.
5. Press **F5** or click **Run BlindAssistantVC.exe** to launch the camera and tracking co-pilot.

## Execution Notes
- The C++ port uses **forward-facing camera orientation** (no horizontal flipping) to ensure that left and right voice instructions match real-world physical obstacles.
- Voice guidance runs asynchronously on Windows COM background threads so camera frame rates never drop or freeze during speech.
- To exit the tracking window, press **'q'** or **ESC**.
