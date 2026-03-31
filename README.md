# Algorithms

## 高光谱推扫影像条带去除（GDAL + OpenCV）

当前实现按你给出的流程组织：

1. **GDAL读影像**（多波段读入 `CV_32F`）。
2. **坏线检测（Dead Line）**：检测行/列标准差异常低的线。
3. **条带方向检测（水平/垂直）**：在均值影像上比较行剖面与列剖面的高频能量。
4. **条带强度评估（弱/中/强）**：用全局一致性条带线占比评估。
5. **策略选择**：
   - 弱：统计校正（邻域参考线增益/偏置归一化）
   - 中：统计校正 + 掩膜线平滑
   - 强：统计校正 + 平滑 + 频域抑制
6. **多波段一致性处理**：按 `--consensus` 汇聚每个波段的检测结果。
7. **GDAL写出**（保留 GeoTransform / Projection）。

## 编译

```bash
g++ -std=c++17 src/stripe_destriping.cpp -o stripe_destriping \
  $(gdal-config --cflags) $(pkg-config --cflags --libs opencv4) \
  $(gdal-config --libs)
```

## 使用

```bash
./stripe_destriping input.tif output.tif \
  --smooth 31 \
  --z 3.0 \
  --consensus 0.2 \
  --dead-std 0.05 \
  --neighbor 3 \
  --weak 0.01 \
  --medium 0.03
```

参数：

- `--smooth`：剖面平滑核（奇数）。
- `--z`：条带 robust z-score 阈值。
- `--consensus`：多波段一致性阈值（0~1）。
- `--dead-std`：坏线标准差阈值（相对全局标准差）。
- `--neighbor`：统计校正时邻域参考线半径。
- `--weak` / `--medium`：条带强度分级阈值（按全局条带线占比）。
