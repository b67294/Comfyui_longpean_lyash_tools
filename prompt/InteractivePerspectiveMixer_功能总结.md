# InteractivePerspectiveMixer 节点功能总结

## 1. 节点定位
`InteractivePerspectiveMixer` 是一个“透视变换 + 图层合成”节点：
- 把 `layer_image`（印花/贴图）按 4 个角点做透视变换；
- 再合成到 `background_image`（底图/材质图）上；
- 支持在前端可视化拖拽角点编辑（Open Perspective Editor）。

适合场景：服装印花、包装贴图、平面图案贴到透视背景等 Mockup 流程。

## 2. 输入与输出

### 必填输入
- `background_image`：背景图（IMAGE）
- `layer_image`：待变换图层（IMAGE）
- `blend_mode`：混合模式
  - `multiply`：乘算混合，结果更贴合底图明暗
  - `normal`：正常覆盖（仅在变换区域生效）
- `invert_bg_mask`：是否反转背景 Alpha

### 可选输入
- `background_mask`（MASK）：背景 Alpha，推荐连接 `LoadImage` 的 MASK 输出
- `layer_mask`（MASK）：图层 Alpha，推荐连接 `LoadImage` 的 MASK 输出
- `corners_input`（STRING）：角点坐标字符串，格式 `[(x,y),(x,y),(x,y),(x,y)]`
  - 顺序固定：左上 TL、右上 TR、右下 BR、左下 BL
  - 值为相对坐标（通常 0~1），允许超界用于消失点效果

### 输出
- `result_image`（IMAGE）：合成结果
- `warped_layer`（IMAGE）：仅透视后的图层（未与背景合成）
- `result_mask`（MASK）：结果遮罩（由背景 Alpha 决定）
- `corners_output`（STRING）：最终生效角点，格式同 `corners_input`

## 3. 核心处理逻辑（后端）

1. 读取背景与图层，转换为 uint8 numpy。
2. 拆分 RGB 与 Alpha：
   - 优先使用外部 `background_mask` / `layer_mask`；
   - 未提供时回退到图像内嵌 alpha（若无则按全不透明处理）。
3. 角点来源优先级：
   - 先解析 `corners_input`；
   - 若无效则自动生成默认居中矩形角点。
4. 用 OpenCV `getPerspectiveTransform + warpPerspective` 进行透视：
   - 分别对 RGB 和 Alpha 进行 warp。
5. 按 `blend_mode` 混合：
   - `multiply`：`(bg * layer) / 255`
   - `normal`：直接用 layer
   - 最终仅在 `warped_alpha` 区域替换背景（其余保留背景）。
6. 输出 Alpha 规则：
   - 输出透明度以背景 alpha 为准（layer 不改变背景 alpha 结构）。
7. 组织输出张量，并返回角点字符串。

## 4. 前端交互能力（interactive_perspective.js）

该节点在 ComfyUI 前端注册了扩展，自动添加按钮：
- `✏ Open Perspective Editor`

编辑器支持：
- 拖拽 4 个角点实时调整透视四边形；
- 拖拽四边形内部移动整体；
- Shift + 拖角点做等比缩放；
- 鼠标滚轮缩放视图、空白区域拖拽平移、`Z` 复位视图；
- 取消/应用；应用后写回 `corners_input`。

前端预览机制：
- 背景图显示在画布；
- 图层通过 CSS `matrix3d` 做实时透视预览；
- 后端会保存临时预览图，供节点 `node.imgs` 在前端读取。

## 5. 使用建议
- 需要正确透明通道时，尽量把 `LoadImage` 的 `MASK` 接到 `background_mask` 和 `layer_mask`。
- 当要做远近透视/消失点效果时，可让角点坐标超出 0~1。
- 若 `corners_input` 提供了有效值，它会覆盖默认角点，并作为实际变换依据。

## 6. 一句话总结
这是一个把“可视化四点透视编辑”与“图层混合合成”整合到同一节点的工具，既能手动交互调形变，也能通过字符串角点实现可复现的参数化流程。