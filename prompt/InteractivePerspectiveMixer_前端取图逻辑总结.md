# InteractivePerspectiveMixer 前端如何获取 background_image 和 layer_image

## 结论先说
前端编辑器拿图有两条路径，按优先级依次尝试：
1. 从当前节点上游已连接的加载节点里，读取 widget 值拼成可访问 URL。
2. 如果上游路径拿不到，再从节点执行后回传的 node.imgs 中读取缓存图 URL。

对应关系：
- background_image: 先查输入名 background_image，再回退到 node.imgs[1]
- layer_image: 先查输入名 layer_image，再回退到 node.imgs[2]

## 1. 前端入口位置
主要逻辑在以下文件：
- [web/js/interactive_perspective.js](web/js/interactive_perspective.js)

核心函数：
- getLoadImageUrl(node, inputName)
- getNodeImgUrl(node, idx)
- openEditor(node) 中的 bgUrl / layerUrl 选择逻辑

## 2. 第一路径：从上游连接节点取 URL
在打开编辑器时，前端会先调用：
- getLoadImageUrl(node, "background_image")
- getLoadImageUrl(node, "layer_image")

这个函数的做法：
1. 在当前节点 inputs 里按名称找输入口。
2. 通过 input.link 拿到图连线，再定位到上游源节点。
3. 遍历上游节点 widgets，按规则推断图片地址：
   - 若发现 URL 类字段（widget 名含 url），直接用其值（http/https/data URL）。
   - 若发现本地文件字段（image/filename/file/path），拼成 /view?filename=...&type=input。
   - 若节点名字看起来像 LoadImage，再做一次 image 类字段兜底。
4. 找不到就返回 null。

说明：这条路径主要解决“节点还没执行过，但上游已经选了图”的情况。

## 3. 第二路径：从 node.imgs 回退
如果第 1 路径失败，前端会回退到：
- background 用 getNodeImgUrl(node, 1)
- layer 用 getNodeImgUrl(node, 2)

getNodeImgUrl 支持三种 node.imgs 元素形式：
- HTMLImageElement: 取 .src
- string: 直接作为 URL
- object: 取 .src 或 .url

## 4. node.imgs 是怎么来的（后端配合）
后端执行节点时会在返回值里附带 ui.images，顺序固定为：
1. result 预览图（node.imgs[0]）
2. background 输入图（node.imgs[1]）
3. layer 输入图（node.imgs[2]）

相关实现位置：
- [src/visual_perspectivemixer/nodes.py](src/visual_perspectivemixer/nodes.py)

后端会把三张图存到 temp 目录并返回 ComfyUI 图片字典，前端再通过 node.imgs 使用这些地址。

## 5. 最终加载到编辑器中的流程
在 openEditor 里：
1. 先计算 bgUrl / layerUrl（先上游，后 node.imgs）。
2. 分别创建两个 Image 对象并设置 crossOrigin="anonymous"。
3. onload 成功后记录 naturalWidth / naturalHeight，用于画布缩放和透视预览。
4. 若某张图失败：
   - background 失败时用棋盘格占位。
   - layer 失败时只是不显示图层预览。

## 6. 一句话记忆
它不是只靠 node.imgs，也不是只靠上游输入，而是“上游直取优先，执行缓存兜底”的双通道取图方案。

## 7. 当前问题（Bug）总结
1. 当前前端 `getLoadImageUrl` 主要面向 `LoadImage/LoadImageFromURL` 这类“有 URL 或文件名 widget”的节点。
2. 当 `background_image` 或 `layer_image` 上游是 `Resize`、混合、蒙版等预处理节点时，通常拿不到可直接访问的 URL。
3. 此时只能依赖 `node.imgs` 兜底；如果节点还没执行过、或没有成功写入 `ui.images`，编辑器就可能拿不到图（背景棋盘格、图层不显示）。
4. 结果表现为：复杂链路下首次打开编辑器预览不稳定，常需要先手动执行一次工作流。

## 8. 修改策略 Prompt（仅本策略）
下面这段可直接给代码修改助手使用：

```text
目标：仅修复 InteractivePerspectiveMixer 编辑器在“上游是预处理节点”时首次取图不稳定的问题。

必须遵守：
1) 只做最小改动，不改节点功能逻辑。
2) 不要做递归溯源，不要向上多层追踪输入来源。
3) 不要新增复杂架构，不要改 UI 交互行为。

改动范围：
- 后端：两个版本都要改
  - InteractivePerspectiveMixer（nodes.py）
  - InteractivePerspectiveMixerAdvanced（nodes_advanced.py）
- 前端：interactive_perspective.js

具体修改要求：
1) 后端两个版本都必须确保每次执行稳定返回 ui.images（至少包含 result/background/layer 三张预览图），以便前端可通过 node.imgs 回退。
2) 前端保持现有两段式策略：
   - 第一段：getLoadImageUrl(node, inputName)（仅当前一层上游节点判断）
   - 第二段：失败后回退 getNodeImgUrl(node, idx)
3) 前端仅增强健壮性：
   - 对 getLoadImageUrl 失败路径增加更清晰日志
   - 对 node.imgs 为空时给出明确提示
   - 不改变现有索引约定（bg=1, layer=2）

验收标准：
1) 上游是 LoadImage：可直接预览。
2) 上游是 Resize 等预处理节点：首次可能无图，但执行一次后可稳定从 node.imgs 预览。
3) 不引入递归查找逻辑。
4) 不影响当前 Open Editor 的交互能力。
```

补充说明：
- 本阶段先完成文档与策略对齐，不直接改代码。
- 代码实施阶段必须同步修改两个后端版本，避免一边可预览、一边不可预览的行为分裂。