#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

struct SpectrumData {
    Mat logMagNorm;   // 用于显示
    Mat power;        // 真实功率谱（中心化后）
};

// =======================
// 频谱中心化
// =======================
void fftShift(Mat& img)
{
    img = img(Rect(0, 0, img.cols & -2, img.rows & -2));
    int cx = img.cols / 2;
    int cy = img.rows / 2;

    Mat q0(img, Rect(0, 0, cx, cy));
    Mat q1(img, Rect(cx, 0, cx, cy));
    Mat q2(img, Rect(0, cy, cx, cy));
    Mat q3(img, Rect(cx, cy, cx, cy));

    Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

static float percentileFromMat(const Mat& src, float p)
{
    CV_Assert(src.type() == CV_32F);
    vector<float> vals;
    vals.reserve(src.total());
    for (int r = 0; r < src.rows; ++r) {
        const float* ptr = src.ptr<float>(r);
        for (int c = 0; c < src.cols; ++c) vals.push_back(ptr[c]);
    }
    if (vals.empty()) return 0.f;

    size_t k = static_cast<size_t>(std::clamp(p, 0.f, 1.f) * (vals.size() - 1));
    nth_element(vals.begin(), vals.begin() + k, vals.end());
    return vals[k];
}

// =======================
// 计算频谱
// =======================
SpectrumData computeSpectrum(const Mat& imgGray)
{
    Mat imgFloat;
    imgGray.convertTo(imgFloat, CV_32F);
    imgFloat -= mean(imgFloat)[0];

    int m = getOptimalDFTSize(imgFloat.rows);
    int n = getOptimalDFTSize(imgFloat.cols);

    Mat padded;
    copyMakeBorder(imgFloat, padded,
                   0, m - imgFloat.rows,
                   0, n - imgFloat.cols,
                   BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {padded, Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg);
    dft(complexImg, complexImg);

    split(complexImg, planes);
    Mat mag;
    magnitude(planes[0], planes[1], mag);

    Mat power;
    pow(mag, 2.0, power);
    fftShift(power);

    // 用于显示
    Mat logMag = power.clone();
    logMag += 1.0f;
    log(logMag, logMag);
    normalize(logMag, logMag, 0, 1, NORM_MINMAX);

    return {logMag, power};
}

// =======================
// 条纹检测核心函数（抑制地物纹理误检）
// =======================
void detectStripe(const Mat& power)
{
    CV_Assert(power.type() == CV_32F);

    Mat energy = power.clone();
    Point2f center(energy.cols / 2.0f, energy.rows / 2.0f);
    float maxRadius = std::min(energy.cols, energy.rows) * 0.5f;

    // 1) 环形带通：去低频(地物大结构) + 去近Nyquist噪声
    float rLow = maxRadius * 0.06f;
    float rHigh = maxRadius * 0.92f;
    for (int y = 0; y < energy.rows; ++y) {
        float* ptr = energy.ptr<float>(y);
        for (int x = 0; x < energy.cols; ++x) {
            float dx = x - center.x;
            float dy = y - center.y;
            float r = std::sqrt(dx * dx + dy * dy);
            if (r < rLow || r > rHigh) ptr[x] = 0.0f;
        }
    }

    // 2) 极坐标变换
    Mat polar;
    warpPolar(energy, polar,
              Size(360, static_cast<int>(maxRadius)),
              center,
              maxRadius,
              WARP_POLAR_LINEAR + WARP_FILL_OUTLIERS);

    // 3) 角度能量（鲁棒截断，减少纹理宽带能量影响）
    Mat angleEnergy;
    reduce(polar, angleEnergy, 0, REDUCE_SUM, CV_32F); // 1x360
    GaussianBlur(angleEnergy, angleEnergy, Size(11, 1), 2.5);

    float p95 = percentileFromMat(angleEnergy, 0.95f);
    if (p95 > 0) {
        threshold(angleEnergy, angleEnergy, p95, p95, THRESH_TRUNC);
    }

    Scalar meanS, stdS;
    meanStdDev(angleEnergy, meanS, stdS);
    float meanVal = static_cast<float>(meanS[0]);
    float stdVal = static_cast<float>(stdS[0]) + 1e-6f;

    double maxVal;
    Point maxLoc;
    minMaxLoc(angleEnergy, nullptr, &maxVal, nullptr, &maxLoc);
    int thetaF = maxLoc.x;

    // 4) 峰尖锐度（地物纹理通常是宽峰）
    float half = static_cast<float>((maxVal + meanVal) * 0.5);
    int l = thetaF, r = thetaF;
    for (int i = 0; i < 180; ++i) {
        int nl = (l - 1 + 360) % 360;
        if (angleEnergy.at<float>(0, nl) < half) break;
        l = nl;
    }
    for (int i = 0; i < 180; ++i) {
        int nr = (r + 1) % 360;
        if (angleEnergy.at<float>(0, nr) < half) break;
        r = nr;
    }
    int widthDeg = (r >= l) ? (r - l + 1) : (r + 360 - l + 1);

    // 5) 角向集中度
    int win = 6;
    float localSum = 0.f, globalSum = 0.f;
    for (int k = 0; k < 360; ++k) {
        float v = angleEnergy.at<float>(0, k);
        globalSum += v;
        int d = std::min((k - thetaF + 360) % 360, (thetaF - k + 360) % 360);
        if (d <= win) localSum += v;
    }
    float concentration = localSum / (globalSum + 1e-6f);

    // 6) 频率方向上的径向峰（要求有明显窄带峰）
    Mat radial = polar.col(thetaF).clone();
    radial.rowRange(0, static_cast<int>(rLow) + 2).setTo(0);

    Mat radialSmooth;
    GaussianBlur(radial, radialSmooth, Size(1, 15), 3.0);

    double rMaxVal;
    Point rLoc;
    minMaxLoc(radialSmooth, nullptr, &rMaxVal, nullptr, &rLoc);
    int rPeak = rLoc.y;

    float radialMean = static_cast<float>(mean(radialSmooth)[0]);
    float radialProm = static_cast<float>(rMaxVal / (radialMean + 1e-6f));

    // 判据：强度、显著性、尖锐度、集中度、径向突出度
    float zScore = static_cast<float>((maxVal - meanVal) / stdVal);
    bool stripeLike =
        (zScore >= 4.0f) &&
        (maxVal / (meanVal + 1e-6f) >= 2.2f) &&
        (widthDeg <= 24) &&
        (concentration >= 0.18f) &&
        (radialProm >= 2.0f) &&
        (rPeak > rLow + 2);

    if (!stripeLike) {
        cout << "未检测到明确的周期性条纹（已抑制地物纹理误检）" << endl;
        return;
    }

    int thetaStripe = thetaF - 90;
    if (thetaStripe < 0) thetaStripe += 180;

    double stripeSpacing = static_cast<double>(energy.cols) / rPeak;

    cout << "======================" << endl;
    cout << "检测到周期性条纹" << endl;
    cout << "方向显著性 Z = " << zScore << endl;
    cout << "角向峰宽(半高) = " << widthDeg << " 度" << endl;
    cout << "角向集中度 = " << concentration << endl;
    cout << "条纹方向 = " << thetaStripe << " 度" << endl;
    cout << "条纹间距 ≈ " << stripeSpacing << " 像素" << endl;
    cout << "======================" << endl;
}

// =======================
// 主函数
// =======================
int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "用法: ./StripeDetect image.tif" << endl;
        return -1;
    }

    Mat img = imread(argv[1], IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "无法读取图像" << endl;
        return -1;
    }

    SpectrumData spec = computeSpectrum(img);
    imshow("Frequency Spectrum", spec.logMagNorm);

    detectStripe(spec.power);

    waitKey(0);
    return 0;
}
