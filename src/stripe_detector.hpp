#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

struct StripeDetectionResult {
    cv::Mat stripeMask;              // 8-bit mask, 255 indicates stripe pixels
    cv::Mat periodicComponent;       // float image: extracted periodic stripe signal
    cv::Mat repairedImage;           // 8-bit repaired image
    cv::Mat spectrumVisualization;   // 8-bit spectrum for debugging
    std::vector<cv::Point> peaks;    // spectrum peak coordinates
    double orientationDeg = 0.0;     // normal direction of stripe in degree
    double periodPixels = 0.0;       // estimated stripe period in pixel
    double periodicConfidence = 0.0; // [0, 1]
};

class PeriodicStripeDetector {
public:
    struct Params {
        int peakSuppressionRadius = 10;      // remove low frequency around center
        double peakThresholdSigma = 2.8;     // threshold = mean + sigma * std in log spectrum
        int maxPeaks = 30;                   // keep strongest peaks for analysis
        double orientationToleranceDeg = 12; // peak-to-line angular tolerance
        double confidenceThreshold = 0.25;   // whether periodic stripe exists
        int localEnergyWindow = 15;          // odd window for local energy
        double maskStdFactor = 0.7;          // local mask threshold scale
        int residualInpaintRadius = 3;       // inpaint radius for unremoved residual
    };

    explicit PeriodicStripeDetector(Params params = Params{});
    StripeDetectionResult process(const cv::Mat& inputGray) const;

private:
    Params params_;

    static cv::Mat toFloat01(const cv::Mat& gray);
    static cv::Mat fftShift(const cv::Mat& src);
    static cv::Mat computeMagnitudeLog(const cv::Mat& complexSpectrum);
    static cv::Mat makeComplexFromReal(const cv::Mat& real);

    std::vector<cv::Point> findSpectrumPeaks(const cv::Mat& logMag) const;
    static double angleDeg(const cv::Point2f& v);
    static double circularDistanceDeg(double a, double b);

    void estimatePeriodicity(
        const std::vector<cv::Point>& peaks,
        cv::Size spectrumSize,
        double* outOrientationDeg,
        double* outPeriodPixels,
        double* outConfidence,
        std::vector<cv::Point>* inlierPeaks) const;

    cv::Mat buildFrequencyMask(cv::Size size, const std::vector<cv::Point>& stripePeaks) const;
    cv::Mat reconstructPeriodicComponent(const cv::Mat& spectrum, const cv::Mat& bandMask) const;
    cv::Mat buildSpatialStripeMask(const cv::Mat& periodicComponent) const;
    cv::Mat repairByNotchAndInpaint(
        const cv::Mat& originalFloat,
        const cv::Mat& spectrum,
        const cv::Mat& stripeMask,
        const cv::Mat& stripeFreqMask) const;
};
