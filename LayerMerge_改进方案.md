# LayerMerge 节点输出格式改进方案

## 1. 现在的问题

### 当前输出格式
`LayerMerge` 节点现在遵循 ComfyUI 标准做法，将图像和透明度通道**分离输出**：

| 输出端口 | 格式 | 说明 |
|---------|------|------|
| `merged_image` | IMAGE (1, H, W, 3) | RGB 3通道，**不含 Alpha** |
| `merged_alpha` | MASK (1, H, W) | 单独的透明度通道 |

### 问题所在
- 下游节点如果要保存或处理完整的 RGBA 图像，需要**手动连接两个输出端口**
- 某些节点可能不支持这种分离格式，或者支持不完善
- 对于直观的RGBA工作流，需要额外节点来重新合并

---

## 2. Alpha 计算逻辑

### RGB 颜色合成
使用**标准 Over 操作**（不预乘 Alpha）：
$$\text{RGB}_{\text{result}} = \text{RGB}_{\text{bg}} \times (1 - \alpha_{\text{layer}}) + \text{RGB}_{\text{layer}} \times \alpha_{\text{layer}}$$

### Alpha 通道合成
使用**取最大值**规则：
$$\alpha_{\text{result}} = \max(\alpha_{\text{bg}}, \alpha_{\text{layer}})$$

这意味着：
- 初始化 `result_alpha = alpha_group1`
- 依次处理后续各组，每一步都取当前累积 alpha 和新图层 alpha 的**较大值**
- 结果是：**只要任何一层在某像素不透明，该像素就是不透明的**

### 数值范围
- RGB: float32, 范围 [0, 1]
- Alpha: float32 (MASK), 范围 [0, 1]，其中 1 = 完全不透明, 0 = 完全透明

---

## 3. 改进方案

### 核心思路
增加一个 **可选开关参数** `output_rgba`，用户可选择输出格式：

| 参数值 | 输出格式 | 说明 |
|-------|---------|------|
| `True` (默认) | RGBA (1, H, W, 4) | 合并后的透明 RGBA 图像 |
| `False` | RGB (1, H, W, 3) | 仅 RGB，分离 MASK 输出 |

### 改进内容

#### INPUT_TYPES 新增
```
"optional": {
    "output_rgba": ("BOOLEAN", {
        "default": True,
        "tooltip": "输出格式：开启则输出 RGBA (含透明通道)，关闭则输出 RGB + 分离 MASK"
    }),
    ...
}
```

#### RETURN_TYPES 动态调整
- 当 `output_rgba=True`:  
  `RETURN_TYPES = ("IMAGE", "MASK")`  
  `RETURN_NAMES = ("merged_image_rgba", "merged_alpha")`  
  其中 `merged_image_rgba` 是 (1, H, W, 4) RGBA

- 当 `output_rgba=False`:  
  `RETURN_TYPES = ("IMAGE", "MASK")`  
  `RETURN_NAMES = ("merged_image", "merged_alpha")`  
  其中 `merged_image` 是 (1, H, W, 3) RGB （保持现状）

**注意**: 两种情况下的 `RETURN_TYPES` 都是 `("IMAGE", "MASK")`，但输出内容不同

#### merge() 方法变更
在最终输出转换部分，根据 `output_rgba` 参数决定：

```python
# 生成最终 RGBA 还是仅 RGB
if output_rgba:
    # 合并 RGB + Alpha 为 RGBA
    result_rgba = np.concatenate(
        [result_rgb_uint8, result_alpha_uint8[:, :, np.newaxis]], 
        axis=-1
    )  # (H, W, 4) uint8
    merged_image_tensor = self._np_to_tensor_rgba(result_rgba)  # 需要新方法
else:
    # 保持现状：仅 RGB
    merged_image_tensor = self._np_to_tensor(result_rgb_uint8)  # (H, W, 3)

# Alpha 输出保持不变
merged_alpha_tensor = self._mask_np_to_tensor(result_alpha_uint8)

return (merged_image_tensor, merged_alpha_tensor)
```

#### 新增辅助方法
```python
@staticmethod
def _np_to_tensor_rgba(arr: np.ndarray) -> torch.Tensor:
    """(H, W, 4) uint8 numpy → (1, H, W, 4) float tensor (RGBA 格式)."""
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
```

---

## 4. 向后兼容性

- 默认 `output_rgba=True`，即**新行为**（输出 RGBA）
- 现有工作流如果依赖旧的 `(RGB, MASK)` 分离格式，需要设置 `output_rgba=False`
- 两种格式都能保证 Alpha 信息不丢失

---

## 5. 使用示例

### 场景 A: 需要完整 RGBA（新默认行为）
```
LayerMerge (output_rgba=True)
  ↓ merged_image_rgba (包含 RGBA, 可直接保存为 PNG)
  ↓ merged_alpha (作为辅助参考)
SaveImage 或其他 RGBA 感知节点
```

### 场景 B: 保持 ComfyUI 标准格式（旧行为）
```
LayerMerge (output_rgba=False)
  ↓ merged_image (RGB 3通道)
  ↓ merged_alpha (MASK)
连接到标准 ComfyUI 工作流
```

---

## 6. 总结

| 项目 | 说明 |
|-----|------|
| **问题** | 输出分离 (RGB+MASK)，某些场景需要合并 RGBA |
| **解决方案** | 增加 `output_rgba` 开关，支持直接输出透明 RGBA |
| **默认行为** | 输出 RGBA（output_rgba=True） |
| **兼容性** | 可通过设置开关回到旧行为（output_rgba=False） |
| **Alpha 计算** | `α = max(α_bg, α_layer)`（取最大值保证透明度正确） |
