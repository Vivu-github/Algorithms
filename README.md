# Algorithms

## 周期性遥感影像条纹检测与修复（OpenCV / C++）

本工程提供一个可直接编译的 C++ 示例：
- **检测**：在频域中寻找明显的对称峰值，并通过**方向一致性**与**频率聚集性**判断是否为周期条纹（区别于一般地物纹理）。
- **定位**：重建条纹周期分量并转到空间域，输出条纹掩膜。
- **修复**：先做频域陷波抑制周期条纹，再对残余区域做 inpaint（Telea）。

> 适用场景：扫描条带、行噪声、推扫成像中的周期性条纹。  
> 非周期且方向杂乱的纹理通常不会通过“周期置信度”判别。

---

## 方法说明

### 1) 周期条纹检测（区分地物纹理）

1. 对灰度影像做 DFT，计算对数幅度频谱。
2. 抑制中心低频后提取局部峰值（候选频率）。
3. 统计峰值方向直方图，找主方向。
4. 计算主方向内点比例（`periodicConfidence`）：
   - 主方向聚集明显 → 更可能是条纹噪声。
   - 峰值方向分散 → 更可能是地物纹理。
5. 依据主峰半径估计周期（pixel）。

### 2) 条纹区域定位

- 仅保留条纹相关频峰并反变换，得到周期分量。
- 对周期分量做局部能量阈值 + 形态学，得到 `stripe_mask.png`。

### 3) 条纹修复

- 频域对条纹峰做平滑陷波，重建去条纹图像。
- 对检测掩膜区域再做 inpaint，处理残余条纹和局部伪影。

---

## 工程结构

```text
.
├── CMakeLists.txt
└── src
    ├── main.cpp
    ├── stripe_detector.hpp
    └── stripe_detector.cpp
```

---

## 编译

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

---

## 运行

```bash
./periodic_stripe_tool <input_image> <output_dir> [--confidence 0.25] [--sigma 2.8]
```

示例：

```bash
./periodic_stripe_tool ../data/test.png ../output --confidence 0.22 --sigma 2.5
```

输出文件：
- `repaired.png`：修复结果
- `stripe_mask.png`：条纹掩膜
- `periodic_component.png`：提取到的周期条纹分量
- `spectrum.png`：频谱可视化

控制参数：
- `--confidence`：周期置信度阈值（越大越保守）
- `--sigma`：频谱峰值阈值强度（越大越只保留强峰）

---

## 可扩展建议

- 多光谱/高光谱：逐波段检测并在掩膜层面做时空一致性融合。
- 大幅面：按块检测后进行周期参数平滑，避免边界跳变。
- 实际生产可加入：
  - 方向先验（如已知沿行/列条纹）
  - 自适应 notch 宽度
  - 条纹强度评价指标（修复前后对比）
