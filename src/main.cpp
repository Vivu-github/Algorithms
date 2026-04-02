#include "stripe_detector.hpp"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace {
void printUsage(const char* program) {
    std::cout << "Usage:\n  " << program
              << " <input_image> <output_dir> [--confidence <value>] [--sigma <value>]\n\n"
              << "Outputs:\n"
              << "  repaired.png          修复后的影像\n"
              << "  stripe_mask.png       条纹空间掩膜\n"
              << "  periodic_component.png 周期条纹分量可视化\n"
              << "  spectrum.png          频谱图\n";
}

cv::Mat visualizePeriodic(const cv::Mat& periodic) {
    cv::Mat absPeriodic;
    cv::absdiff(periodic, cv::Scalar(0), absPeriodic);
    cv::Mat normalized;
    cv::normalize(absPeriodic, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    return normalized;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::filesystem::path inputPath = argv[1];
    std::filesystem::path outDir = argv[2];

    PeriodicStripeDetector::Params params;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--confidence") {
            params.confidenceThreshold = std::stod(value);
        } else if (key == "--sigma") {
            params.peakThresholdSigma = std::stod(value);
        } else {
            std::cerr << "Unknown option: " << key << "\n";
            printUsage(argv[0]);
            return 2;
        }
    }

    const cv::Mat image = cv::imread(inputPath.string(), cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to read input image: " << inputPath << "\n";
        return 3;
    }

    std::filesystem::create_directories(outDir);

    PeriodicStripeDetector detector(params);
    const StripeDetectionResult result = detector.process(image);

    cv::imwrite((outDir / "repaired.png").string(), result.repairedImage);
    cv::imwrite((outDir / "stripe_mask.png").string(), result.stripeMask);
    cv::imwrite((outDir / "periodic_component.png").string(), visualizePeriodic(result.periodicComponent));
    cv::imwrite((outDir / "spectrum.png").string(), result.spectrumVisualization);

    std::cout << "Periodic stripe detection done." << std::endl;
    std::cout << "  confidence    : " << result.periodicConfidence << std::endl;
    std::cout << "  orientation   : " << result.orientationDeg << " deg" << std::endl;
    std::cout << "  period        : " << result.periodPixels << " px" << std::endl;
    std::cout << "Outputs saved to: " << outDir << std::endl;

    return 0;
}
