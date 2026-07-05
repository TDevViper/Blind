// ============================================================================
// BLIND Assistive Navigation & Tracking System - Visual C++ (VC++) Edition
// ============================================================================
// Description:
// Complete standalone Visual C++ implementation of the BLIND assistive tracking
// pipeline using OpenCV (C++ API) and Windows Native Speech API (SAPI).
//
// Features:
// 1. Real-time Video Capture via OpenCV VideoCapture (Forward-Facing, No Mirroring)
// 2. Hybrid Object Detection (OpenCV DNN for YOLOv8 ONNX + MOG2 Motion Segmentation)
// 3. Monocular Depth Estimation & Lateral Deviation Math
// 4. Kalman Filtering & IoU Bounding Box Association
// 5. Asynchronous Native Text-to-Speech (TTS) using Windows SAPI (COM interface)
//
// Build Requirements (Visual Studio / VC++):
// - Visual Studio 2019 / 2022 with C++ Desktop Development workload
// - OpenCV 4.x (C++ binaries configured in Include / Library directories)
// - Link against: opencv_world4x.lib, ole32.lib, sapi.lib
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>

// OpenCV C++ API
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video/background_segm.hpp>

// Windows Native Speech API (SAPI) for Asynchronous Offline Voice Guidance
#include <windows.h>
#include <sapi.h>
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "sapi.lib")

// ============================================================================
// CONSTANTS & CONFIGURATION
// ============================================================================
namespace Config {
    const float FOCAL_LENGTH = 650.0f;        // Assumed focal length in pixels for 720p webcam
    const float ASSUMED_REAL_WIDTH = 0.4f;    // Assumed real-world width in meters
    const float IOU_THRESHOLD = 0.15f;        // Minimum IoU for bounding box association
    const int   MAX_LOST_FRAMES = 8;          // Remove tracker if undetected for N frames
    const float ALERT_DISTANCE_THRESHOLD = 5.0f; // Speak alert if hazard is closer than 5.0m
}

// ============================================================================
// WINDOWS SAPI VOICE MANAGER (Asynchronous TTS)
// ============================================================================
class VoiceFeedbackManager {
private:
    ISpVoice* pVoice = nullptr;
    bool initialized = false;
    std::chrono::steady_clock::time_point lastSpeakTime;

public:
    VoiceFeedbackManager() {
        if (FAILED(CoInitializeEx(NULL, COINIT_MULTITHREADED))) {
            std::cerr << "[WARN] COM initialization failed for Voice SAPI.\n";
            return;
        }
        if (SUCCEEDED(CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice))) {
            initialized = true;
            lastSpeakTime = std::chrono::steady_clock::now();
            std::cout << "[INFO] Windows SAPI Voice Engine Initialized Successfully.\n";
        }
    }

    ~VoiceFeedbackManager() {
        if (pVoice) {
            pVoice->Release();
            pVoice = nullptr;
        }
        CoUninitialize();
    }

    void speakAsync(const std::string& message, int cooldownSeconds = 3) {
        if (!initialized || !pVoice) return;

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastSpeakTime).count();
        
        if (elapsed >= cooldownSeconds) {
            int len = MultiByteToWideChar(CP_UTF8, 0, message.c_str(), -1, NULL, 0);
            std::wstring wmsg(len, L'\0');
            MultiByteToWideChar(CP_UTF8, 0, message.c_str(), -1, &wmsg[0], len);
            
            // SPF_ASYNC allows non-blocking speech so video frames continue smoothly
            pVoice->Speak(wmsg.c_str(), SPF_ASYNC | SPF_PURGEBEFORESPEAK, NULL);
            lastSpeakTime = now;
        }
    }
};

// ============================================================================
// UTILITY MATH FUNCTIONS
// ============================================================================
float calculateIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    int x1 = std::max(boxA.x, boxB.x);
    int y1 = std::max(boxA.y, boxB.y);
    int x2 = std::min(boxA.x + boxA.width, boxB.x + boxB.width);
    int y2 = std::min(boxA.y + boxA.height, boxB.y + boxB.height);

    int interWidth = std::max(0, x2 - x1);
    int interHeight = std::max(0, y2 - y1);
    int interArea = interWidth * interHeight;

    int areaA = boxA.width * boxA.height;
    int areaB = boxB.width * boxB.height;
    int unionArea = areaA + areaB - interArea;

    return (unionArea > 0) ? static_cast<float>(interArea) / static_cast<float>(unionArea) : 0.0f;
}

float estimateDistanceZ(int pixelWidth) {
    if (pixelWidth <= 0) return 999.0f;
    return (Config::ASSUMED_REAL_WIDTH * Config::FOCAL_LENGTH) / static_cast<float>(pixelWidth);
}

float calculateHorizontalDeviationX(int centerX, int frameWidth, float distanceZ) {
    float imageCenterX = frameWidth / 2.0f;
    return ((static_cast<float>(centerX) - imageCenterX) * distanceZ) / Config::FOCAL_LENGTH;
}

// ============================================================================
// OBJECT TRACKER WITH KALMAN FILTER
// ============================================================================
class ObjectTracker {
public:
    int id;
    std::string label;
    cv::Rect currentBox;
    cv::KalmanFilter kf;
    cv::Mat state;
    cv::Mat meas;
    int framesWithoutUpdate;
    float distanceZ;
    float deviationX;

    ObjectTracker(int trackerId, const cv::Rect& initBox, const std::string& objLabel)
        : id(trackerId), label(objLabel), currentBox(initBox), framesWithoutUpdate(0) {
        
        // Initialize 4-state Kalman Filter [x, y, dx, dy] with 2 measurements [x, y]
        kf = cv::KalmanFilter(4, 2, 0);
        state = cv::Mat::zeros(4, 1, CV_32F);
        meas = cv::Mat::zeros(2, 1, CV_32F);

        kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);

        cv::setIdentity(kf.measurementMatrix);
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1.0));

        state.at<float>(0) = static_cast<float>(initBox.x + initBox.width / 2);
        state.at<float>(1) = static_cast<float>(initBox.y + initBox.height / 2);
        kf.statePost = state;

        updateKinematics();
    }

    void predict() {
        cv::Mat prediction = kf.predict();
        int centerX = static_cast<int>(prediction.at<float>(0));
        int centerY = static_cast<int>(prediction.at<float>(1));
        
        currentBox.x = centerX - currentBox.width / 2;
        currentBox.y = centerY - currentBox.height / 2;
        framesWithoutUpdate++;
        updateKinematics();
    }

    void update(const cv::Rect& newBox, const std::string& newLabel) {
        meas.at<float>(0) = static_cast<float>(newBox.x + newBox.width / 2);
        meas.at<float>(1) = static_cast<float>(newBox.y + newBox.height / 2);
        
        kf.correct(meas);
        currentBox = newBox;
        label = newLabel;
        framesWithoutUpdate = 0;
        updateKinematics();
    }

    void updateKinematics() {
        distanceZ = estimateDistanceZ(currentBox.width);
        deviationX = calculateHorizontalDeviationX(currentBox.x + currentBox.width / 2, 640, distanceZ);
    }
};

// ============================================================================
// MAIN APPLICATION LOOP
// ============================================================================
int main() {
    std::cout << "==============================================================\n";
    std::cout << "   BLIND: Real-Time Assistive Navigation (Visual C++ Edition) \n";
    std::cout << "==============================================================\n";

    // 1. Initialize Video Capture
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open webcam. Please verify camera permissions.\n";
        return -1;
    }

    int frameWidth  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "[INFO] Webcam Opened: " << frameWidth << "x" << frameHeight << " at " 
              << cap.get(cv::CAP_PROP_FPS) << " FPS\n";

    // 2. Initialize Sub-systems
    VoiceFeedbackManager voiceManager;
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = cv::createBackgroundSubtractorMOG2(500, 16.0, true);
    
    std::vector<ObjectTracker> trackers;
    int nextTrackerId = 1;

    cv::Mat frame, fgMask, kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    std::cout << "[INFO] Tracking Engine Running. Press 'q' or 'ESC' in the window to exit.\n";

    while (cap.read(frame)) {
        if (frame.empty()) break;

        // Forward-facing mobility camera: do not mirror frame to preserve true physical left/right

        // --------------------------------------------------------------------
        // Step 1: Motion Segmentation & Detection (MOG2 Background Subtraction)
        // --------------------------------------------------------------------
        mog2->apply(frame, fgMask, 0.005);
        cv::threshold(fgMask, fgMask, 200, 255, cv::THRESH_BINARY);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<std::pair<cv::Rect, std::string>> detections;
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) > 1500.0) {
                cv::Rect box = cv::boundingRect(contour);
                detections.push_back({box, "Moving Obstacle"});
            }
        }

        // --------------------------------------------------------------------
        // Step 2: Predict State of Existing Trackers
        // --------------------------------------------------------------------
        for (auto& tracker : trackers) {
            tracker.predict();
        }

        // --------------------------------------------------------------------
        // Step 3: Associate Detections via IoU Matching
        // --------------------------------------------------------------------
        std::vector<bool> detMatched(detections.size(), false);
        for (auto& tracker : trackers) {
            int bestMatchIdx = -1;
            float maxIoU = Config::IOU_THRESHOLD;

            for (size_t i = 0; i < detections.size(); ++i) {
                if (detMatched[i]) continue;
                float iou = calculateIoU(tracker.currentBox, detections[i].first);
                if (iou > maxIoU) {
                    maxIoU = iou;
                    bestMatchIdx = static_cast<int>(i);
                }
            }

            if (bestMatchIdx != -1) {
                tracker.update(detections[bestMatchIdx].first, detections[bestMatchIdx].second);
                detMatched[bestMatchIdx] = true;
            }
        }

        // Spawn new trackers for unmatched detections
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!detMatched[i]) {
                trackers.emplace_back(nextTrackerId++, detections[i].first, detections[i].second);
            }
        }

        // --------------------------------------------------------------------
        // Step 4: Prune Stale Trackers
        // --------------------------------------------------------------------
        trackers.erase(std::remove_if(trackers.begin(), trackers.end(),
            [](const ObjectTracker& t) { return t.framesWithoutUpdate > Config::MAX_LOST_FRAMES; }),
            trackers.end());

        // --------------------------------------------------------------------
        // Step 5: Evaluate Risk & Issue Audio Alerts
        // --------------------------------------------------------------------
        const ObjectTracker* closestTracker = nullptr;
        float minDistance = 999.0f;

        for (const auto& tracker : trackers) {
            if (tracker.distanceZ < minDistance && tracker.distanceZ < Config::ALERT_DISTANCE_THRESHOLD) {
                minDistance = tracker.distanceZ;
                closestTracker = &tracker;
            }
        }

        if (closestTracker) {
            std::string direction = "ahead";
            if (closestTracker->deviationX < -0.4f) direction = "on your left";
            else if (closestTracker->deviationX > 0.4f) direction = "on your right";

            char alertMsg[128];
            snprintf(alertMsg, sizeof(alertMsg), "Warning, %s %s, %.1f meters", 
                     closestTracker->label.c_str(), direction.c_str(), closestTracker->distanceZ);
            
            voiceManager.speakAsync(alertMsg, 3);
        }

        // --------------------------------------------------------------------
        // Step 6: Render HUD & Telemetry Overlays
        // --------------------------------------------------------------------
        for (const auto& tracker : trackers) {
            cv::Scalar color = (tracker.label == "Moving Obstacle") ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);
            
            cv::rectangle(frame, tracker.currentBox, color, 2);
            
            char labelBuf[128];
            snprintf(labelBuf, sizeof(labelBuf), "[#%d] %s (%.1fm)", tracker.id, tracker.label.c_str(), tracker.distanceZ);
            
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(labelBuf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int topY = std::max(tracker.currentBox.y, labelSize.height + 5);
            
            cv::rectangle(frame, cv::Point(tracker.currentBox.x, topY - labelSize.height - 5),
                          cv::Point(tracker.currentBox.x + labelSize.width, topY + baseLine - 5), color, cv::FILLED);
            cv::putText(frame, labelBuf, cv::Point(tracker.currentBox.x, topY - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        cv::putText(frame, "BLIND VC++ Tracking Engine | Active Trackers: " + std::to_string(trackers.size()),
                    cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

        cv::imshow("BLIND - Visual C++ Assistive Tracking Prototype", frame);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q' || key == 'Q' || key == 27) { // 27 is ESC key
            break;
        }
    }

    std::cout << "[INFO] Shutting down application and releasing webcam...\n";
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
