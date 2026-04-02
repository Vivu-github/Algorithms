#ifndef STRIPE_DETECTION_HPP
#define STRIPE_DETECTION_HPP

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace stripe {

struct StripeDetectionResult {
    std::string axis;               // "row" or "col"
    std::vector<int> indices;       // detected stripe row/col indices
    std::vector<double> scores;     // robust z-score per row/col
    cv::Mat mask;                   // CV_8U mask: stripe=255, normal=0
};

struct RepairResult {
    cv::Mat repairedImage;          // same size/type as input
    cv::Mat binaryMask01;           // CV_8U mask: repair=1, normal=0
};

struct AngleStripeDetectionResult {
    bool hasStripe = false;
    double bestAngleDeg = 0.0;      // best stripe angle in degrees
    double bestScore = 0.0;         // max abs robust z-score across scanned angles
};

StripeDetectionResult detectRowStripes(
    const cv::Mat& image,
    double threshold = 3.5,
    const std::string& reducer = "mean",
    int minRun = 1,
    bool useParallel = true);

StripeDetectionResult detectColStripes(
    const cv::Mat& image,
    double threshold = 3.5,
    const std::string& reducer = "mean",
    int minRun = 1,
    bool useParallel = true);

RepairResult repairStripeRegionLinear(
    const cv::Mat& image,
    const cv::Mat& stripeMask255,
    int maxRadius = 10,
    bool useParallel = true);

RepairResult detectAndRepairStripesLinear(
    const cv::Mat& image,
    double threshold = 3.5,
    const std::string& reducer = "mean",
    int minRun = 1,
    int maxRadius = 10,
    bool useParallel = true);

bool hasStripes(
    const cv::Mat& image,
    double threshold = 3.5,
    const std::string& reducer = "mean",
    int minRun = 1,
    bool useParallel = true);

AngleStripeDetectionResult detectAnyAngleStripes(
    const cv::Mat& image,
    double threshold = 3.5,
    const std::string& reducer = "mean",
    int minRun = 1,
    double angleMinDeg = 0.0,
    double angleMaxDeg = 180.0,
    double angleStepDeg = 2.0,
    bool useParallel = true);

RepairResult detectAndRepairAnyAngleStripes(
    const cv::Mat& image,
    double threshold = 3.5,
    const std::string& reducer = "mean",
    int minRun = 1,
    int maxRadius = 10,
    double angleMinDeg = 0.0,
    double angleMaxDeg = 180.0,
    double angleStepDeg = 2.0,
    bool useParallel = true);

} // namespace stripe

#endif // STRIPE_DETECTION_HPP
