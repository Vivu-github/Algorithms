# Algorithms

## 条带检测与修复（工程化：`stripe_detection.hpp` + `stripe_detection.cpp`）

已重构为工程级结构：

- `stripe_detection.hpp`：对外接口声明（可直接被你的工程 `#include`）
- `stripe_detection.cpp`：完整实现（检测 + 修复）

## API

命名空间：`stripe`

- `detectRowStripes(...)`
- `detectColStripes(...)`
- `repairStripeRegionLinear(...)`
- `detectAndRepairStripesLinear(...)`
- `hasStripes(...)`（仅判断是否有条纹：有返回 `true`，无返回 `false`）
- `detectAnyAngleStripes(...)`（检测任意角度斜条纹，返回是否存在 + 最可能角度）
- `detectAndRepairAnyAngleStripes(...)`（任意角度检测并修复）
  
以上接口新增 `useParallel` 参数（默认 `true`），可启用 OpenCV 的 `parallel_for_` 做分块并行加速。

结果结构：

- `StripeDetectionResult`
  - `axis` / `indices` / `scores` / `mask`
- `RepairResult`
  - `repairedImage` / `binaryMask01`

## 能力说明

- 条带检测：行/列统计 + MAD 鲁棒 z-score + 阈值 + `minRun`
- 条带修复：按掩膜区域进行四邻域线性插值 + 边缘镜像扩展 + 修复区域平滑
- 并行加速：按行/列分块并行执行统计与修复计算（可通过 `useParallel=false` 关闭）
- 任意角度斜条纹检测：通过角度扫描 + 旋转投影（`warpAffine` + 行统计）实现
- 默认角度范围已改为全角度扫描：`0° ~ 180°`（可通过参数自定义）
- 支持多通道逐通道修复
- 支持 8 位、16 位、浮点数据；修复后按原类型范围裁剪并回写

## 编译（作为库源文件）

```bash
g++ -std=c++17 -c stripe_detection.cpp `pkg-config --cflags opencv4`
```

在你的可执行工程里，把 `stripe_detection.cpp` 一起编译并链接 OpenCV 即可。

## 最小调用示例

```cpp
#include "stripe_detection.hpp"
#include <opencv2/imgcodecs.hpp>

int main() {
    cv::Mat img = cv::imread("input.tif", cv::IMREAD_UNCHANGED);
    auto angleRes = stripe::detectAnyAngleStripes(img, 3.0, "mean", 2); // 默认 0~180 全角度
    if (!angleRes.hasStripe) return 0;
    auto result = stripe::detectAndRepairAnyAngleStripes(img, 3.0, "mean", 2, 12); // 默认 0~180 全角度检测+修复
    cv::imwrite("repaired.tif", result.repairedImage);
    return 0;
}
```
