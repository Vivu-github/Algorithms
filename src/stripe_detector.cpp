#include "stripe_detector.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {
constexpr double kPi = 3.14159265358979323846;

cv::Mat normalizeTo8U(const cv::Mat& src) {
    cv::Mat normalized;
    cv::normalize(src, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    return normalized;
}

cv::Mat makeSymmetric(const cv::Mat& mask) {
    cv::Mat flipped;
    cv::flip(mask, flipped, -1);
    return cv::max(mask, flipped);
}
} // namespace

PeriodicStripeDetector::PeriodicStripeDetector(Params params) : params_(std::move(params)) {}

cv::Mat PeriodicStripeDetector::toFloat01(const cv::Mat& gray) {
    cv::Mat grayFloat;
    if (gray.type() == CV_32F) {
        grayFloat = gray.clone();
    } else {
        gray.convertTo(grayFloat, CV_32F, 1.0 / 255.0);
    }
    return grayFloat;
}

cv::Mat PeriodicStripeDetector::fftShift(const cv::Mat& src) {
    cv::Mat shifted = src.clone();
    const int cx = shifted.cols / 2;
    const int cy = shifted.rows / 2;

    cv::Mat q0(shifted, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(shifted, cv::Rect(cx, 0, shifted.cols - cx, cy));
    cv::Mat q2(shifted, cv::Rect(0, cy, cx, shifted.rows - cy));
    cv::Mat q3(shifted, cv::Rect(cx, cy, shifted.cols - cx, shifted.rows - cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    return shifted;
}

cv::Mat PeriodicStripeDetector::computeMagnitudeLog(const cv::Mat& complexSpectrum) {
    std::vector<cv::Mat> channels(2);
    cv::split(complexSpectrum, channels);
    cv::Mat magnitude;
    cv::magnitude(channels[0], channels[1], magnitude);
    magnitude += 1.0f;
    cv::log(magnitude, magnitude);
    return magnitude;
}

cv::Mat PeriodicStripeDetector::makeComplexFromReal(const cv::Mat& real) {
    cv::Mat imag = cv::Mat::zeros(real.size(), CV_32F);
    std::vector<cv::Mat> planes = {real, imag};
    cv::Mat complex;
    cv::merge(planes, complex);
    return complex;
}

std::vector<cv::Point> PeriodicStripeDetector::findSpectrumPeaks(const cv::Mat& logMag) const {
    const cv::Point center(logMag.cols / 2, logMag.rows / 2);
    cv::Mat filtered = logMag.clone();

    cv::circle(filtered, center, params_.peakSuppressionRadius, cv::Scalar(0), cv::FILLED);

    cv::Scalar mean, stddev;
    cv::meanStdDev(filtered, mean, stddev);
    const double threshold = mean[0] + params_.peakThresholdSigma * stddev[0];

    cv::Mat dilated;
    cv::dilate(filtered, dilated, cv::Mat());
    cv::Mat localMax = (filtered == dilated) & (filtered > threshold);

    std::vector<cv::Point> candidates;
    cv::findNonZero(localMax, candidates);

    std::sort(candidates.begin(), candidates.end(), [&](const cv::Point& a, const cv::Point& b) {
        return filtered.at<float>(a) > filtered.at<float>(b);
    });

    if (static_cast<int>(candidates.size()) > params_.maxPeaks) {
        candidates.resize(params_.maxPeaks);
    }
    return candidates;
}

double PeriodicStripeDetector::angleDeg(const cv::Point2f& v) {
    double a = std::atan2(v.y, v.x) * 180.0 / kPi;
    if (a < 0.0) {
        a += 180.0;
    }
    return a;
}

double PeriodicStripeDetector::circularDistanceDeg(double a, double b) {
    double d = std::fabs(a - b);
    return std::min(d, 180.0 - d);
}

void PeriodicStripeDetector::estimatePeriodicity(
    const std::vector<cv::Point>& peaks,
    cv::Size spectrumSize,
    double* outOrientationDeg,
    double* outPeriodPixels,
    double* outConfidence,
    std::vector<cv::Point>* inlierPeaks) const {

    const cv::Point2f center(static_cast<float>(spectrumSize.width) / 2.0f,
                             static_cast<float>(spectrumSize.height) / 2.0f);

    if (peaks.empty()) {
        *outOrientationDeg = 0.0;
        *outPeriodPixels = 0.0;
        *outConfidence = 0.0;
        inlierPeaks->clear();
        return;
    }

    std::vector<double> angles;
    angles.reserve(peaks.size());
    for (const auto& p : peaks) {
        cv::Point2f v = cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)) - center;
        if (cv::norm(v) < 1e-3f) {
            continue;
        }
        angles.push_back(angleDeg(v));
    }

    if (angles.empty()) {
        *outOrientationDeg = 0.0;
        *outPeriodPixels = 0.0;
        *outConfidence = 0.0;
        inlierPeaks->clear();
        return;
    }

    constexpr int kBins = 180;
    std::vector<int> histogram(kBins, 0);
    for (double a : angles) {
        const int idx = std::clamp(static_cast<int>(std::round(a)) % kBins, 0, kBins - 1);
        histogram[idx] += 1;
    }

    const int dominantIdx = static_cast<int>(std::distance(
        histogram.begin(), std::max_element(histogram.begin(), histogram.end())));
    const double dominantAngle = static_cast<double>(dominantIdx);

    inlierPeaks->clear();
    std::vector<double> frequencies;
    frequencies.reserve(peaks.size());

    for (const auto& p : peaks) {
        cv::Point2f v = cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)) - center;
        const double radius = cv::norm(v);
        if (radius < 1e-3) {
            continue;
        }
        const double a = angleDeg(v);
        if (circularDistanceDeg(a, dominantAngle) <= params_.orientationToleranceDeg) {
            inlierPeaks->push_back(p);
            frequencies.push_back(radius / static_cast<double>(std::min(spectrumSize.width, spectrumSize.height)));
        }
    }

    if (inlierPeaks->size() < 2) {
        *outOrientationDeg = dominantAngle;
        *outPeriodPixels = 0.0;
        *outConfidence = static_cast<double>(inlierPeaks->size()) / std::max<size_t>(1, peaks.size());
        return;
    }

    std::sort(frequencies.begin(), frequencies.end());
    const double medianFrequency = frequencies[frequencies.size() / 2];
    const double period = (medianFrequency > 1e-6) ? (1.0 / medianFrequency) : 0.0;

    *outOrientationDeg = dominantAngle;
    *outPeriodPixels = period;
    *outConfidence = static_cast<double>(inlierPeaks->size()) / static_cast<double>(peaks.size());
}

cv::Mat PeriodicStripeDetector::buildFrequencyMask(cv::Size size, const std::vector<cv::Point>& stripePeaks) const {
    cv::Mat mask(size, CV_32F, cv::Scalar(0));
    for (const auto& peak : stripePeaks) {
        cv::circle(mask, peak, 4, cv::Scalar(1.0f), cv::FILLED);
    }
    mask = makeSymmetric(mask);

    cv::GaussianBlur(mask, mask, cv::Size(0, 0), 2.0);
    cv::normalize(mask, mask, 0.0, 1.0, cv::NORM_MINMAX);
    return mask;
}

cv::Mat PeriodicStripeDetector::reconstructPeriodicComponent(
    const cv::Mat& spectrum,
    const cv::Mat& bandMask) const {

    cv::Mat shiftedSpectrum = fftShift(spectrum);

    std::vector<cv::Mat> channels(2);
    cv::split(shiftedSpectrum, channels);
    channels[0] = channels[0].mul(bandMask);
    channels[1] = channels[1].mul(bandMask);
    cv::merge(channels, shiftedSpectrum);

    cv::Mat unshifted = fftShift(shiftedSpectrum);
    cv::Mat periodicComplex;
    cv::dft(unshifted, periodicComplex, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    return periodicComplex;
}

cv::Mat PeriodicStripeDetector::buildSpatialStripeMask(const cv::Mat& periodicComponent) const {
    cv::Mat absPeriodic;
    cv::absdiff(periodicComponent, cv::Scalar(0), absPeriodic);

    cv::Mat localMean, localSqMean;
    cv::blur(absPeriodic, localMean, cv::Size(params_.localEnergyWindow, params_.localEnergyWindow));

    cv::Mat sq;
    cv::multiply(absPeriodic, absPeriodic, sq);
    cv::blur(sq, localSqMean, cv::Size(params_.localEnergyWindow, params_.localEnergyWindow));

    cv::Mat localStd;
    cv::sqrt(cv::max(localSqMean - localMean.mul(localMean), 0), localStd);

    cv::Scalar globalMean, globalStd;
    cv::meanStdDev(absPeriodic, globalMean, globalStd);
    const double threshold = globalMean[0] + params_.maskStdFactor * globalStd[0];

    cv::Mat mask = absPeriodic > threshold;
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

    mask.convertTo(mask, CV_8U, 255);
    return mask;
}

cv::Mat PeriodicStripeDetector::repairByNotchAndInpaint(
    const cv::Mat& originalFloat,
    const cv::Mat& spectrum,
    const cv::Mat& stripeMask,
    const cv::Mat& stripeFreqMask) const {

    cv::Mat shifted = fftShift(spectrum);
    cv::Mat notch = 1.0f - stripeFreqMask;

    std::vector<cv::Mat> channels(2);
    cv::split(shifted, channels);
    channels[0] = channels[0].mul(notch);
    channels[1] = channels[1].mul(notch);
    cv::merge(channels, shifted);

    cv::Mat filteredComplex = fftShift(shifted);
    cv::Mat filtered;
    cv::dft(filteredComplex, filtered, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    cv::Mat filtered8u;
    cv::Mat clipped = cv::min(cv::max(filtered, 0.0f), 1.0f);
    clipped.convertTo(filtered8u, CV_8U, 255.0);

    cv::Mat src8u;
    originalFloat.convertTo(src8u, CV_8U, 255.0);

    cv::Mat inpaintMask;
    cv::threshold(stripeMask, inpaintMask, 0, 255, cv::THRESH_BINARY);

    cv::Mat inpainted;
    cv::inpaint(filtered8u, inpaintMask, inpainted, params_.residualInpaintRadius, cv::INPAINT_TELEA);
    return inpainted;
}

StripeDetectionResult PeriodicStripeDetector::process(const cv::Mat& inputGray) const {
    CV_Assert(!inputGray.empty());
    CV_Assert(inputGray.channels() == 1);

    StripeDetectionResult result;
    const cv::Mat grayFloat = toFloat01(inputGray);

    cv::Mat complexSpectrum;
    cv::dft(makeComplexFromReal(grayFloat), complexSpectrum);

    cv::Mat logMag = computeMagnitudeLog(fftShift(complexSpectrum));
    result.spectrumVisualization = normalizeTo8U(logMag);

    const auto peaks = findSpectrumPeaks(logMag);
    result.peaks = peaks;

    std::vector<cv::Point> stripePeaks;
    estimatePeriodicity(peaks,
                        logMag.size(),
                        &result.orientationDeg,
                        &result.periodPixels,
                        &result.periodicConfidence,
                        &stripePeaks);

    if (result.periodicConfidence < params_.confidenceThreshold || stripePeaks.size() < 2) {
        result.stripeMask = cv::Mat::zeros(inputGray.size(), CV_8U);
        result.periodicComponent = cv::Mat::zeros(inputGray.size(), CV_32F);
        result.repairedImage = inputGray.clone();
        return result;
    }

    cv::Mat stripeFreqMask = buildFrequencyMask(logMag.size(), stripePeaks);
    result.periodicComponent = reconstructPeriodicComponent(complexSpectrum, stripeFreqMask);
    result.stripeMask = buildSpatialStripeMask(result.periodicComponent);
    result.repairedImage = repairByNotchAndInpaint(grayFloat, complexSpectrum, result.stripeMask, stripeFreqMask);

    return result;
}
