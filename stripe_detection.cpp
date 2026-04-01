#include "stripe_detection.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

namespace stripe {
namespace {

struct NeighborInfo {
    bool found = false;
    int r = -1;
    int c = -1;
    double value = 0.0;
};

void getTypeRange(int depth, double& minV, double& maxV) {
    switch (depth) {
        case CV_8U:  minV = 0.0; maxV = 255.0; break;
        case CV_8S:  minV = -128.0; maxV = 127.0; break;
        case CV_16U: minV = 0.0; maxV = 65535.0; break;
        case CV_16S: minV = -32768.0; maxV = 32767.0; break;
        case CV_32S: minV = static_cast<double>(INT32_MIN); maxV = static_cast<double>(INT32_MAX); break;
        case CV_32F: minV = -3.4e38; maxV = 3.4e38; break;
        case CV_64F: minV = -1.7e308; maxV = 1.7e308; break;
        default:
            throw std::invalid_argument("Unsupported image depth");
    }
}

cv::Mat clampAndConvertLikeInput(const cv::Mat& srcF64, int dstType) {
    double minV = 0.0;
    double maxV = 0.0;
    getTypeRange(CV_MAT_DEPTH(dstType), minV, maxV);

    cv::Mat clipped;
    cv::max(srcF64, minV, clipped);
    cv::min(clipped, maxV, clipped);

    cv::Mat dst;
    clipped.convertTo(dst, dstType);
    return dst;
}

double computeMedian(std::vector<double> v) {
    if (v.empty()) {
        throw std::invalid_argument("Input for median is empty");
    }
    const size_t n = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double med = v[mid];
    if (n % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
        med = 0.5 * (med + v[mid - 1]);
    }
    return med;
}

cv::Mat toGrayF64(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        image.convertTo(gray, CV_64F);
    } else {
        cv::Mat gray8;
        cv::cvtColor(image, gray8, cv::COLOR_BGR2GRAY);
        gray8.convertTo(gray, CV_64F);
    }
    return gray;
}

std::vector<double> robustZScore(const std::vector<double>& values, double eps = 1e-9) {
    double med = computeMedian(values);
    std::vector<double> absDev(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        absDev[i] = std::abs(values[i] - med);
    }
    double mad = computeMedian(absDev);
    double scale = 1.4826 * mad + eps;

    std::vector<double> z(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        z[i] = (values[i] - med) / scale;
    }
    return z;
}

std::vector<bool> keepMinRuns(const std::vector<bool>& flags, int minRun) {
    if (minRun <= 1) return flags;

    std::vector<bool> out(flags.size(), false);
    size_t i = 0;
    while (i < flags.size()) {
        if (!flags[i]) {
            ++i;
            continue;
        }
        size_t j = i + 1;
        while (j < flags.size() && flags[j]) ++j;
        if (static_cast<int>(j - i) >= minRun) {
            for (size_t k = i; k < j; ++k) out[k] = true;
        }
        i = j;
    }
    return out;
}

std::vector<double> reduceRows(const cv::Mat& gray, bool useMedian, bool useParallel) {
    std::vector<double> profile(gray.rows, 0.0);
    auto body = [&](const cv::Range& range) {
        for (int r = range.start; r < range.end; ++r) {
            std::vector<double> row(gray.cols);
            for (int c = 0; c < gray.cols; ++c) row[c] = gray.at<double>(r, c);

            if (useMedian) {
                profile[r] = computeMedian(row);
            } else {
                profile[r] = std::accumulate(row.begin(), row.end(), 0.0) / static_cast<double>(gray.cols);
            }
        }
    };
    if (useParallel) {
        cv::parallel_for_(cv::Range(0, gray.rows), body);
    } else {
        body(cv::Range(0, gray.rows));
    }
    return profile;
}

std::vector<double> reduceCols(const cv::Mat& gray, bool useMedian, bool useParallel) {
    std::vector<double> profile(gray.cols, 0.0);
    auto body = [&](const cv::Range& range) {
        for (int c = range.start; c < range.end; ++c) {
            std::vector<double> col(gray.rows);
            for (int r = 0; r < gray.rows; ++r) col[r] = gray.at<double>(r, c);

            if (useMedian) {
                profile[c] = computeMedian(col);
            } else {
                profile[c] = std::accumulate(col.begin(), col.end(), 0.0) / static_cast<double>(gray.rows);
            }
        }
    };
    if (useParallel) {
        cv::parallel_for_(cv::Range(0, gray.cols), body);
    } else {
        body(cv::Range(0, gray.cols));
    }
    return profile;
}

inline bool inBounds(int r, int c, int rows, int cols) {
    return r >= 0 && r < rows && c >= 0 && c < cols;
}

int mirrorIndex(int idx, int length) {
    if (length <= 1) return 0;
    while (idx < 0 || idx >= length) {
        if (idx < 0) {
            idx = -idx - 1;
        } else {
            idx = 2 * length - idx - 1;
        }
    }
    return idx;
}

NeighborInfo findNearestNormal(const cv::Mat& channelF64, const cv::Mat& repairMask01, int r, int c, int dr, int dc, int maxRadius) {
    const int rows = channelF64.rows;
    const int cols = channelF64.cols;

    for (int step = 1; step <= maxRadius; ++step) {
        int nr = r + dr * step;
        int nc = c + dc * step;
        if (!inBounds(nr, nc, rows, cols)) break;
        if (repairMask01.at<uchar>(nr, nc) == 0) {
            return {true, nr, nc, channelF64.at<double>(nr, nc)};
        }
    }

    for (int step = 1; step <= maxRadius; ++step) {
        int nr = mirrorIndex(r + dr * step, rows);
        int nc = mirrorIndex(c + dc * step, cols);
        if (repairMask01.at<uchar>(nr, nc) == 0) {
            return {true, nr, nc, channelF64.at<double>(nr, nc)};
        }
    }

    return {};
}

double interpolate1D(int x1, double y1, int x2, double y2, int x) {
    if (x1 == x2) return 0.5 * (y1 + y2);
    double m = (y2 - y1) / static_cast<double>(x2 - x1);
    double b = y1 - m * static_cast<double>(x1);
    return m * static_cast<double>(x) + b;
}

cv::Mat repairSingleChannelByLinearInterpolation(const cv::Mat& srcChannel, const cv::Mat& repairMask01, int maxRadius, bool useParallel) {
    cv::Mat channelF64;
    srcChannel.convertTo(channelF64, CV_64F);
    cv::Mat repaired = channelF64.clone();

    auto body = [&](const cv::Range& range) {
        for (int r = range.start; r < range.end; ++r) {
            for (int c = 0; c < channelF64.cols; ++c) {
                if (repairMask01.at<uchar>(r, c) == 0) continue;

                NeighborInfo up = findNearestNormal(channelF64, repairMask01, r, c, -1, 0, maxRadius);
                NeighborInfo down = findNearestNormal(channelF64, repairMask01, r, c, 1, 0, maxRadius);
                NeighborInfo left = findNearestNormal(channelF64, repairMask01, r, c, 0, -1, maxRadius);
                NeighborInfo right = findNearestNormal(channelF64, repairMask01, r, c, 0, 1, maxRadius);

                std::vector<double> estimates;
                if (up.found && down.found) {
                    estimates.push_back(interpolate1D(up.r, up.value, down.r, down.value, r));
                }
                if (left.found && right.found) {
                    estimates.push_back(interpolate1D(left.c, left.value, right.c, right.value, c));
                }

                if (estimates.empty()) {
                    if (up.found) estimates.push_back(up.value);
                    if (down.found) estimates.push_back(down.value);
                    if (left.found) estimates.push_back(left.value);
                    if (right.found) estimates.push_back(right.value);
                }

                if (!estimates.empty()) {
                    double value = std::accumulate(estimates.begin(), estimates.end(), 0.0) /
                                   static_cast<double>(estimates.size());
                    repaired.at<double>(r, c) = value;
                }
            }
        }
    };
    if (useParallel) {
        cv::parallel_for_(cv::Range(0, channelF64.rows), body);
    } else {
        body(cv::Range(0, channelF64.rows));
    }

    cv::Mat smoothed;
    cv::GaussianBlur(repaired, smoothed, cv::Size(3, 3), 0.8);
    for (int r = 0; r < repaired.rows; ++r) {
        for (int c = 0; c < repaired.cols; ++c) {
            if (repairMask01.at<uchar>(r, c) != 0) {
                repaired.at<double>(r, c) = smoothed.at<double>(r, c);
            }
        }
    }

    return repaired;
}

} // namespace

StripeDetectionResult detectRowStripes(const cv::Mat& image, double threshold, const std::string& reducer, int minRun, bool useParallel) {
    cv::Mat gray = toGrayF64(image);
    bool useMedian = (reducer == "median");
    if (!(reducer == "mean" || reducer == "median")) {
        throw std::invalid_argument("reducer must be 'mean' or 'median'");
    }

    std::vector<double> profile = reduceRows(gray, useMedian, useParallel);
    std::vector<double> scores = robustZScore(profile);

    std::vector<bool> flags(scores.size(), false);
    for (size_t i = 0; i < scores.size(); ++i) flags[i] = std::abs(scores[i]) >= threshold;
    flags = keepMinRuns(flags, minRun);

    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8U);
    std::vector<int> indices;
    for (int r = 0; r < gray.rows; ++r) {
        if (flags[r]) {
            indices.push_back(r);
            mask.row(r).setTo(255);
        }
    }

    return {"row", indices, scores, mask};
}

StripeDetectionResult detectColStripes(const cv::Mat& image, double threshold, const std::string& reducer, int minRun, bool useParallel) {
    cv::Mat gray = toGrayF64(image);
    bool useMedian = (reducer == "median");
    if (!(reducer == "mean" || reducer == "median")) {
        throw std::invalid_argument("reducer must be 'mean' or 'median'");
    }

    std::vector<double> profile = reduceCols(gray, useMedian, useParallel);
    std::vector<double> scores = robustZScore(profile);

    std::vector<bool> flags(scores.size(), false);
    for (size_t i = 0; i < scores.size(); ++i) flags[i] = std::abs(scores[i]) >= threshold;
    flags = keepMinRuns(flags, minRun);

    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8U);
    std::vector<int> indices;
    for (int c = 0; c < gray.cols; ++c) {
        if (flags[c]) {
            indices.push_back(c);
            mask.col(c).setTo(255);
        }
    }

    return {"col", indices, scores, mask};
}

RepairResult repairStripeRegionLinear(const cv::Mat& image, const cv::Mat& stripeMask255, int maxRadius, bool useParallel) {
    if (image.empty()) {
        throw std::invalid_argument("image is empty");
    }
    if (stripeMask255.empty() || stripeMask255.size() != image.size() || stripeMask255.type() != CV_8U) {
        throw std::invalid_argument("stripeMask255 must be CV_8U and same size as image");
    }

    cv::Mat repairMask01;
    cv::threshold(stripeMask255, repairMask01, 0, 1, cv::THRESH_BINARY);

    cv::Mat repaired;
    if (image.channels() == 1) {
        cv::Mat repairedF64 = repairSingleChannelByLinearInterpolation(image, repairMask01, maxRadius, useParallel);
        repaired = clampAndConvertLikeInput(repairedF64, image.type());
    } else {
        std::vector<cv::Mat> srcChannels;
        cv::split(image, srcChannels);

        std::vector<cv::Mat> outChannels(srcChannels.size());
        for (size_t i = 0; i < srcChannels.size(); ++i) {
            cv::Mat repairedF64 = repairSingleChannelByLinearInterpolation(srcChannels[i], repairMask01, maxRadius, useParallel);
            outChannels[i] = clampAndConvertLikeInput(repairedF64, srcChannels[i].type());
        }
        cv::merge(outChannels, repaired);
    }

    return {repaired, repairMask01};
}

RepairResult detectAndRepairStripesLinear(const cv::Mat& image, double threshold, const std::string& reducer, int minRun, int maxRadius, bool useParallel) {
    StripeDetectionResult rowRes = detectRowStripes(image, threshold, reducer, minRun, useParallel);
    StripeDetectionResult colRes = detectColStripes(image, threshold, reducer, minRun, useParallel);

    cv::Mat mergedMask;
    cv::bitwise_or(rowRes.mask, colRes.mask, mergedMask);

    return repairStripeRegionLinear(image, mergedMask, maxRadius, useParallel);
}

bool hasStripes(const cv::Mat& image, double threshold, const std::string& reducer, int minRun, bool useParallel) {
    StripeDetectionResult rowRes = detectRowStripes(image, threshold, reducer, minRun, useParallel);
    if (!rowRes.indices.empty()) {
        return true;
    }
    StripeDetectionResult colRes = detectColStripes(image, threshold, reducer, minRun, useParallel);
    return !colRes.indices.empty();
}

} // namespace stripe
