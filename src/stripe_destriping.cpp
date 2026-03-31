#include <gdal_priv.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

enum class StripeStrength { Weak, Medium, Strong };

struct Config {
    std::string inputPath;
    std::string outputPath;
    int smoothKernel = 31;             // 条带检测剖面平滑核
    double zThreshold = 3.0;           // robust zscore 阈值
    double consensus = 0.2;            // 多波段一致性
    double deadLineStdRatio = 0.05;    // 坏线标准差阈值（相对全局std）
    int neighborRadius = 3;            // 统计校正参考邻域
    double weakRatio = 0.01;           // 条带强度阈值：弱
    double mediumRatio = 0.03;         // 条带强度阈值：中
};

static void ensureOdd(int& v) {
    if (v < 1) v = 1;
    if ((v & 1) == 0) ++v;
}

static cv::Mat gdalBandToMat32F(GDALRasterBand* band) {
    const int w = band->GetXSize();
    const int h = band->GetYSize();
    cv::Mat m(h, w, CV_32F);
    if (band->RasterIO(GF_Read, 0, 0, w, h, m.data, w, h, GDT_Float32, 0, 0) != CE_None) {
        throw std::runtime_error("GDAL RasterIO read failed.");
    }
    return m;
}

static void mat32FToGdalBand(const cv::Mat& m, GDALRasterBand* band) {
    const int w = band->GetXSize();
    const int h = band->GetYSize();
    if (m.rows != h || m.cols != w || m.type() != CV_32F) throw std::runtime_error("Size/type mismatch.");
    if (band->RasterIO(GF_Write, 0, 0, w, h, const_cast<uchar*>(m.data), w, h, GDT_Float32, 0, 0) != CE_None) {
        throw std::runtime_error("GDAL RasterIO write failed.");
    }
}

static cv::Mat lineProfile(const cv::Mat& band, bool vertical) {
    cv::Mat p;
    cv::reduce(band, p, vertical ? 0 : 1, cv::REDUCE_AVG, CV_32F);
    if (p.rows == 1) p = p.t();
    return p;
}

static cv::Mat robustDetectLines(const cv::Mat& profile, double zThreshold) {
    std::vector<float> v(profile.rows);
    for (int i = 0; i < profile.rows; ++i) v[i] = profile.at<float>(i, 0);

    std::vector<float> s = v;
    std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
    const float med = s[s.size() / 2];

    std::vector<float> dev(v.size());
    std::transform(v.begin(), v.end(), dev.begin(), [med](float x) { return std::fabs(x - med); });
    std::nth_element(dev.begin(), dev.begin() + dev.size() / 2, dev.end());
    const float mad = dev[dev.size() / 2];
    const float scale = 1.4826f * std::max(mad, 1e-6f);

    cv::Mat mask = cv::Mat::zeros(profile.rows, 1, CV_8U);
    for (int i = 0; i < profile.rows; ++i) {
        float z = std::fabs((v[i] - med) / scale);
        if (z >= zThreshold) mask.at<uchar>(i, 0) = 255;
    }
    return mask;
}

static cv::Mat detectDeadLines(const cv::Mat& band, bool vertical, double deadLineStdRatio) {
    const int n = vertical ? band.cols : band.rows;
    cv::Scalar globalMean, globalStd;
    cv::meanStdDev(band, globalMean, globalStd);
    const double t = std::max(1e-6, globalStd[0] * deadLineStdRatio);

    cv::Mat dead = cv::Mat::zeros(n, 1, CV_8U);
    for (int i = 0; i < n; ++i) {
        cv::Mat line = vertical ? band.col(i) : band.row(i);
        cv::Scalar m, s;
        cv::meanStdDev(line, m, s);
        if (s[0] <= t) dead.at<uchar>(i, 0) = 255;
    }
    return dead;
}

static bool detectStripeOrientation(const cv::Mat& meanBand) {
    // true=vertical, false=horizontal
    cv::Mat colP = lineProfile(meanBand, true);
    cv::Mat rowP = lineProfile(meanBand, false);

    cv::Mat colS, rowS;
    cv::GaussianBlur(colP, colS, cv::Size(1, 31), 0, 0, cv::BORDER_REFLECT);
    cv::GaussianBlur(rowP, rowS, cv::Size(1, 31), 0, 0, cv::BORDER_REFLECT);

    const double colEnergy = cv::norm(colP - colS, cv::NORM_L2);
    const double rowEnergy = cv::norm(rowP - rowS, cv::NORM_L2);
    return colEnergy >= rowEnergy;
}

static cv::Mat consensusMask(const std::vector<cv::Mat>& masks, double consensus) {
    const int n = masks[0].rows;
    const int bands = static_cast<int>(masks.size());
    const int minVotes = std::max(1, static_cast<int>(std::ceil(consensus * bands)));

    cv::Mat out = cv::Mat::zeros(n, 1, CV_8U);
    for (int i = 0; i < n; ++i) {
        int votes = 0;
        for (const auto& m : masks) votes += m.at<uchar>(i, 0) > 0 ? 1 : 0;
        if (votes >= minVotes) out.at<uchar>(i, 0) = 255;
    }
    return out;
}

static double lineRatio(const cv::Mat& m) {
    return m.rows > 0 ? static_cast<double>(cv::countNonZero(m)) / m.rows : 0.0;
}

static StripeStrength assessStrength(double ratio, const Config& cfg) {
    if (ratio < cfg.weakRatio) return StripeStrength::Weak;
    if (ratio < cfg.mediumRatio) return StripeStrength::Medium;
    return StripeStrength::Strong;
}

static std::vector<int> neighborIdx(int i, int n, const cv::Mat& mask, int r) {
    std::vector<int> out;
    for (int d = 1; d <= r; ++d) {
        if (i - d >= 0 && mask.at<uchar>(i - d, 0) == 0) out.push_back(i - d);
        if (i + d < n && mask.at<uchar>(i + d, 0) == 0) out.push_back(i + d);
    }
    return out;
}

static void statisticalCorrect(cv::Mat& band, const cv::Mat& mask, bool vertical, int neighborRadius) {
    cv::Mat work = vertical ? band : band.t();
    for (int c = 0; c < work.cols; ++c) {
        if (mask.at<uchar>(c, 0) == 0) continue;
        auto refs = neighborIdx(c, work.cols, mask, neighborRadius);
        if (refs.empty()) continue;

        cv::Mat stripe = work.col(c).clone();
        cv::Mat ref = cv::Mat::zeros(work.rows, 1, CV_32F);
        for (int k : refs) ref += work.col(k);
        ref /= static_cast<float>(refs.size());

        cv::Scalar mx = cv::mean(ref), my = cv::mean(stripe);
        cv::Mat xc = ref - mx[0], yc = stripe - my[0];
        double varx = std::max(1e-12, static_cast<double>(xc.dot(xc)) / work.rows);
        double cov = static_cast<double>(xc.dot(yc)) / work.rows;
        double a = cov / varx;
        if (!std::isfinite(a) || std::fabs(a) < 1e-6) a = 1.0;
        double b = my[0] - a * mx[0];
        cv::Mat corrected = (stripe - static_cast<float>(b)) / static_cast<float>(a);
        corrected.copyTo(work.col(c));
    }
    band = vertical ? work : work.t();
}

static void smoothOnMaskedLines(cv::Mat& band, const cv::Mat& mask, bool vertical) {
    cv::Mat blur;
    cv::GaussianBlur(band, blur, vertical ? cv::Size(7, 1) : cv::Size(1, 7), 0, 0, cv::BORDER_REFLECT);
    if (vertical) {
        for (int c = 0; c < band.cols; ++c) if (mask.at<uchar>(c, 0) > 0) blur.col(c).copyTo(band.col(c));
    } else {
        for (int r = 0; r < band.rows; ++r) if (mask.at<uchar>(r, 0) > 0) blur.row(r).copyTo(band.row(r));
    }
}

// 强条带：对剖面做1D频域抑制，然后按偏置场扣除
static void frequencySuppressStrong(cv::Mat& band, bool vertical) {
    cv::Mat work = vertical ? band : band.t();
    cv::Mat profile = lineProfile(work, true); // 列均值

    cv::Mat dftIn;
    profile.convertTo(dftIn, CV_32F);
    cv::dft(dftIn, dftIn, cv::DFT_COMPLEX_OUTPUT);

    std::vector<cv::Mat> ch(2);
    cv::split(dftIn, ch);
    cv::Mat mag;
    cv::magnitude(ch[0], ch[1], mag);

    // 抑制高能周期峰（保留低频直流）
    for (int i = 2; i < mag.rows / 2; ++i) {
        if (mag.at<float>(i, 0) > 5.0f * cv::mean(mag)[0]) {
            ch[0].at<float>(i, 0) = 0.0f;
            ch[1].at<float>(i, 0) = 0.0f;
            ch[0].at<float>(mag.rows - i, 0) = 0.0f;
            ch[1].at<float>(mag.rows - i, 0) = 0.0f;
        }
    }

    cv::merge(ch, dftIn);
    cv::Mat filtered;
    cv::dft(dftIn, filtered, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    cv::Mat bias = profile - filtered;
    for (int c = 0; c < work.cols; ++c) {
        work.col(c) -= bias.at<float>(c, 0);
    }
    band = vertical ? work : work.t();
}

static Config parseArgs(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: stripe_destriping <input.tif> <output.tif> [--smooth 31] [--z 3.0]"
            " [--consensus 0.2] [--dead-std 0.05] [--neighbor 3] [--weak 0.01] [--medium 0.03]\n");
    }
    Config cfg;
    cfg.inputPath = argv[1];
    cfg.outputPath = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value: " + a);
            return argv[++i];
        };
        if (a == "--smooth") cfg.smoothKernel = std::stoi(next());
        else if (a == "--z") cfg.zThreshold = std::stod(next());
        else if (a == "--consensus") cfg.consensus = std::stod(next());
        else if (a == "--dead-std") cfg.deadLineStdRatio = std::stod(next());
        else if (a == "--neighbor") cfg.neighborRadius = std::stoi(next());
        else if (a == "--weak") cfg.weakRatio = std::stod(next());
        else if (a == "--medium") cfg.mediumRatio = std::stod(next());
        else throw std::runtime_error("Unknown arg: " + a);
    }

    ensureOdd(cfg.smoothKernel);
    cfg.consensus = std::clamp(cfg.consensus, 0.0, 1.0);
    cfg.neighborRadius = std::max(1, cfg.neighborRadius);
    return cfg;
}

int main(int argc, char** argv) {
    try {
        const Config cfg = parseArgs(argc, argv);
        GDALAllRegister();

        GDALDataset* inDs = static_cast<GDALDataset*>(GDALOpen(cfg.inputPath.c_str(), GA_ReadOnly));
        if (!inDs) throw std::runtime_error("Cannot open input dataset.");

        const int w = inDs->GetRasterXSize();
        const int h = inDs->GetRasterYSize();
        const int bands = inDs->GetRasterCount();
        if (bands <= 0) throw std::runtime_error("No bands in dataset.");

        // GDAL读影像
        std::vector<cv::Mat> cube;
        cube.reserve(bands);
        for (int b = 1; b <= bands; ++b) cube.push_back(gdalBandToMat32F(inDs->GetRasterBand(b)));

        // 条带方向检测（水平/垂直）
        cv::Mat meanBand = cv::Mat::zeros(h, w, CV_32F);
        for (const auto& b : cube) meanBand += b;
        meanBand /= static_cast<float>(bands);
        const bool vertical = detectStripeOrientation(meanBand);

        // 坏线检测 + 条带检测（每个波段）
        std::vector<cv::Mat> allMasks;
        allMasks.reserve(bands);
        for (const auto& b : cube) {
            cv::Mat p = lineProfile(b, vertical);
            cv::Mat smooth;
            cv::GaussianBlur(p, smooth, cv::Size(1, cfg.smoothKernel), 0, 0, cv::BORDER_REFLECT);
            cv::Mat stripeMask = robustDetectLines(p - smooth, cfg.zThreshold);
            cv::Mat deadMask = detectDeadLines(b, vertical, cfg.deadLineStdRatio);
            cv::Mat merged;
            cv::bitwise_or(stripeMask, deadMask, merged);
            allMasks.push_back(merged);
        }

        // 多波段一致性处理
        cv::Mat globalMask = consensusMask(allMasks, cfg.consensus);
        const double ratio = lineRatio(globalMask);
        const StripeStrength strength = assessStrength(ratio, cfg);

        std::cout << "Orientation=" << (vertical ? "vertical" : "horizontal")
                  << ", line_ratio=" << ratio << std::endl;

        // GDAL写出准备
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        if (!driver) throw std::runtime_error("GTiff driver not found.");
        GDALDataset* outDs = driver->Create(cfg.outputPath.c_str(), w, h, bands, GDT_Float32, nullptr);
        if (!outDs) throw std::runtime_error("Cannot create output dataset.");

        double gt[6];
        if (inDs->GetGeoTransform(gt) == CE_None) outDs->SetGeoTransform(gt);
        const char* proj = inDs->GetProjectionRef();
        if (proj && std::strlen(proj) > 0) outDs->SetProjection(proj);

        // 策略选择与修复
        for (int b = 0; b < bands; ++b) {
            cv::Mat out = cube[b].clone();
            if (cv::countNonZero(globalMask) > 0) {
                statisticalCorrect(out, globalMask, vertical, cfg.neighborRadius); // 弱:统计校正
                if (strength == StripeStrength::Medium || strength == StripeStrength::Strong) {
                    smoothOnMaskedLines(out, globalMask, vertical); // 中:统计+平滑
                }
                if (strength == StripeStrength::Strong) {
                    frequencySuppressStrong(out, vertical); // 强:统计+频域
                }
            }
            mat32FToGdalBand(out, outDs->GetRasterBand(b + 1));
        }

        GDALClose(outDs);
        GDALClose(inDs);

        std::cout << "Done: " << cfg.outputPath << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
