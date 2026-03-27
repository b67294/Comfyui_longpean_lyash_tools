from inspect import cleandoc
import json
import os
import uuid
import io
import urllib.request
import numpy as np
import cv2
import torch
from PIL import Image
import folder_paths


class InteractivePerspectiveMixer:
    """
    Interactive Perspective Mixer

    Applies a perspective transform (defined by 4 interactive corner handles in
    the frontend editor) to a layer/print image and composites it on top of a
    background/material image. Supports Normal and Multiply blend modes.

    Corner order (TL → TR → BR → BL, stored as relative 0-1 coordinates).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE", {
                    "tooltip": "Background / base material image (e.g. white mock-up)."
                }),
                "layer_image": ("IMAGE", {
                    "tooltip": "Print / decal image to be perspective-transformed."
                }),
                "blend_mode": (["multiply", "normal"], {
                    "default": "multiply",
                    "tooltip": (
                        "multiply: Result = (Background × Layer) / 255  "
                        "normal: Layer replaces background within the warped area."
                    ),
                }),
                "invert_bg_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "反转背景图的透明度通道："
                        "开启后，原本透明的像素变不透明，原本不透明的像素变透明。"
                    ),
                }),
            },
            "optional": {
                "background_mask": ("MASK", {
                    "tooltip": (
                        "背景图的 Alpha 通道（遮罩）。"
                        "请将 LoadImage 节点的 MASK 输出连接到这里，"
                        "以保证输出 32 位 RGBA PNG 并正确保留透明区域。"
                    ),
                }),
                "layer_mask": ("MASK", {
                    "tooltip": (
                        "印花图的 Alpha 通道（遮罩）。"
                        "请将 LoadImage 节点的 MASK 输出连接到这里，"
                        "以正确处理印花图的透明边缘。"
                    ),
                }),
                "corners_input": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "可选坐标输入，格式为 [(x,y),(x,y),(x,y),(x,y)]（百分比小数）。"
                        "顺序为：左上、右上、右下、左下（顺时针）。"
                        "若输入有效坐标，优先级高于 Open Editor 的坐标，"
                        "直接按此坐标进行透视变换。该字段同时也是 Open Editor 按钮的坐标写入目标。"
                    ),
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES  = ("result_image", "warped_layer", "result_mask", "corners_output")
    FUNCTION      = "apply_perspective"
    CATEGORY      = "ComfyUI_tools_for_longpean_zsy"
    DESCRIPTION   = cleandoc(__doc__)

    # ------------------------------------------------------------------ helpers

    def _default_corners(self, bg_w: int, bg_h: int,
                         layer_w: int, layer_h: int) -> list[list[float]]:
        """Return default relative corners that centre the layer on the BG."""
        scale = min(bg_w / max(layer_w, 1), bg_h / max(layer_h, 1)) * 0.8
        w_rel = (layer_w * scale) / bg_w
        h_rel = (layer_h * scale) / bg_h
        cx, cy = 0.5, 0.5
        x0, y0 = cx - w_rel / 2, cy - h_rel / 2
        x1, y1 = cx + w_rel / 2, cy + h_rel / 2
        # TL, TR, BR, BL
        return [
            {"x": x0, "y": y0},
            {"x": x1, "y": y0},
            {"x": x1, "y": y1},
            {"x": x0, "y": y1},
        ]

    @staticmethod
    def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
        """(1, H, W, C) float tensor → (H, W, C) uint8 numpy."""
        return (t[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _split_alpha(img: np.ndarray):
        """Returns (rgb_uint8, alpha_uint8). Always returns 3-ch RGB + separate alpha."""
        if img.shape[2] == 4:
            return img[:, :, :3], img[:, :, 3]
        return img[:, :, :3], np.full(img.shape[:2], 255, dtype=np.uint8)

    @staticmethod
    def _mask_tensor_to_uint8(m: torch.Tensor) -> np.ndarray:
        """
        ComfyUI MASK shape: (1, H, W) float32, 1=opaque 0=transparent
        → (H, W) uint8
        """
        return (m[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _parse_corners_input(s: str):
        """
        解析 [(x,y),(x,y),(x,y),(x,y)] 格式的坐标字符串。
        返回内部统一表示 [{"x":float, "y":float}, ...]，或 None。
        x/y 为相对坐标（百分比），支持超出 [0,1] 以实现消失点效果。
        """
        import ast
        if not s or not s.strip():
            return None
        try:
            parsed = ast.literal_eval(s.strip())
            if (
                isinstance(parsed, (list, tuple))
                and len(parsed) == 4
                and all(
                    isinstance(pt, (list, tuple)) and len(pt) == 2
                    for pt in parsed
                )
            ):
                return [{"x": float(pt[0]), "y": float(pt[1])} for pt in parsed]
        except Exception:
            pass
        return None

    @staticmethod
    def _corners_to_output_str(corners_list: list) -> str:
        """
        将内部 [{"x":..., "y":...}, ...] 转换为
        [(x,y),(x,y),(x,y),(x,y)] 格式字符串（百分比）。
        """
        pts = [(round(c["x"], 6), round(c["y"], 6)) for c in corners_list]
        return "[" + ",".join(f"({x},{y})" for x, y in pts) + "]"

    # ------------------------------------------------------------------ main

    def apply_perspective(
        self,
        background_image: torch.Tensor,
        layer_image: torch.Tensor,
        blend_mode: str,
        invert_bg_mask: bool = False,
        background_mask: torch.Tensor = None,
        layer_mask: torch.Tensor = None,
        corners_input: str = "",
    ):
        bg_np    = self._tensor_to_uint8(background_image)
        layer_np = self._tensor_to_uint8(layer_image)

        bg_h,    bg_w    = bg_np.shape[:2]
        layer_h, layer_w = layer_np.shape[:2]

        # ---- 提取 RGB（LoadImage 的 IMAGE 输出只有 3 通道）-----------------
        bg_rgb,    _bg_alpha_embedded    = self._split_alpha(bg_np)
        layer_rgb, _layer_alpha_embedded = self._split_alpha(layer_np)

        # ---- Alpha 通道：优先使用外部 MASK 输入 ----------------------------
        # LoadImage 会把 alpha 单独从 MASK 口输出，IMAGE 口只有 RGB
        # 因此必须接 MASK 才能拿到真正的 alpha 信息
        if background_mask is not None:
            bg_alpha     = self._mask_tensor_to_uint8(background_mask)
            output_has_alpha = True
        else:
            # 没连 MASK：回退到图片内嵌 alpha（通常是全 255）
            bg_alpha     = _bg_alpha_embedded
            output_has_alpha = (bg_np.shape[2] == 4)

        if layer_mask is not None:
            layer_alpha = self._mask_tensor_to_uint8(layer_mask)
        else:
            layer_alpha = _layer_alpha_embedded

        # ---- 可选：反转背景遮罩 ----------------------------------------------
        if invert_bg_mask:
            bg_alpha = 255 - bg_alpha

        # ---- 解析角点：优先级 corners_input > 默认 -------------------------
        # 1. 尝试解析 corners_input（[(x,y),...] 格式）
        active_corners = self._parse_corners_input(corners_input)

        # 2. 仍然没有：使用默认居中布局
        if active_corners is None:
            active_corners = self._default_corners(bg_w, bg_h, layer_w, layer_h)

        dst_pts = np.array(
            [[c["x"] * bg_w, c["y"] * bg_h] for c in active_corners],
            dtype=np.float32,
        )

        # ---- perspective transform ------------------------------------------
        # Source: four corners of layer image  (TL, TR, BR, BL)
        src_pts = np.array(
            [
                [0,           0          ],
                [layer_w - 1, 0          ],
                [layer_w - 1, layer_h - 1],
                [0,           layer_h - 1],
            ],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warp_flags  = cv2.INTER_LINEAR
        warp_border = cv2.BORDER_CONSTANT

        warped_rgb   = cv2.warpPerspective(
            layer_rgb, M, (bg_w, bg_h),
            flags=warp_flags, borderMode=warp_border, borderValue=(0, 0, 0),
        )
        warped_alpha = cv2.warpPerspective(
            layer_alpha, M, (bg_w, bg_h),
            flags=warp_flags, borderMode=warp_border, borderValue=0,
        )

        # ---- blending -------------------------------------------------------
        # warped_alpha_normalized: warped alpha of the print (0-1)
        # Note: renamed to avoid shadowing the 'layer_mask' parameter
        warped_alpha_normalized = warped_alpha.astype(np.float32) / 255.0     # (H,W)
        warped_alpha_3ch = warped_alpha_normalized[:, :, np.newaxis]                # broadcast

        bg_f    = bg_rgb.astype(np.float32)
        layer_f = warped_rgb.astype(np.float32)

        if blend_mode == "multiply":
            # Result = (Background × Layer) / 255
            blended = (bg_f * layer_f) / 255.0
        else:  # "normal"
            blended = layer_f

        # RGB 合成：在 layer warped 区域内用 blended 替换 bg，其余保留 bg
        # 使用直通 alpha（Straight Alpha）—— 不预乘，PNG 标准格式
        result_rgb_f = bg_f * (1.0 - warped_alpha_3ch) + blended * warped_alpha_3ch
        result_u8    = result_rgb_f.clip(0, 255).astype(np.uint8)

        # ---- 输出 alpha 通道 ------------------------------------------------
        # 规则：背景透明的地方输出必须透明；背景不透明的地方 alpha = bg_alpha
        # （即输出 alpha 完全由背景 alpha 决定，layer 不影响 alpha）
        # ---- 输出 alpha 通道 ------------------------------------------------
        # 规则：output_alpha = bg_alpha（背景透明处输出一定透明）
        # layer 不会让背景变得更透明或更不透明
        if output_has_alpha:
            result_rgba = np.concatenate(
                [result_u8, bg_alpha[:, :, np.newaxis]], axis=-1
            )
            result_tensor = torch.from_numpy(
                result_rgba.astype(np.float32) / 255.0
            ).unsqueeze(0)
        else:
            result_tensor = torch.from_numpy(
                result_u8.astype(np.float32) / 255.0
            ).unsqueeze(0)

        # MASK 输出：供下游节点使用（shape: 1, H, W）
        result_mask_tensor = torch.from_numpy(
            bg_alpha.astype(np.float32) / 255.0
        ).unsqueeze(0)

        # warped_layer 输出：仅经过透视变换的 layer，无合成
        # 格式与原始 layer 保持一致：如果原 layer 有 alpha 就输出 RGBA，否则 RGB
        # 注意：layer_mask 参数优先级高于内嵌 alpha
        original_layer_has_alpha = (layer_mask is not None) or (layer_np.shape[2] == 4)
        if original_layer_has_alpha:
            warped_layer_u8 = np.concatenate(
                [warped_rgb, warped_alpha[:, :, np.newaxis]], axis=-1
            )
        else:
            warped_layer_u8 = warped_rgb
        warped_layer_tensor = torch.from_numpy(
            warped_layer_u8.astype(np.float32) / 255.0
        ).unsqueeze(0)

        # 坐标输出：[(x,y),(x,y),(x,y),(x,y)] 格式字符串（积极使用的坐标）
        corners_out_str = self._corners_to_output_str(active_corners)

        # Save temp previews so ComfyUI populates node.imgs on the frontend:
        #   node.imgs[0] = result (shown as node output thumbnail)
        #   node.imgs[1] = background input  (used by Open Editor as bg canvas)
        #   node.imgs[2] = layer input       (used by Open Editor as layer overlay)
        # bg_np and layer_np are saved as-is (preserving original channels incl. alpha)
        # so the editor shows the actual source image, not a stripped RGB copy.
        ui_images = []
        try:
            tmpdir = folder_paths.get_temp_directory()
            uid = uuid.uuid4().hex[:10]
            
            def _save(arr, tag):
                """Save numpy array to PNG and return ComfyUI format dict."""
                if arr is None or arr.size == 0:
                    return None
                
                name = f"ipm_{tag}_{uid}.png"
                try:
                    # Ensure array is uint8
                    if arr.dtype != np.uint8:
                        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.dtype in [np.float32, np.float64] else arr.astype(np.uint8)
                    
                    # Determine channels
                    if arr.ndim == 2:
                        # Grayscale
                        pil_img = Image.fromarray(arr, mode="L")
                    elif arr.ndim == 3:
                        c = arr.shape[2]
                        if c == 1:
                            pil_img = Image.fromarray(arr[:, :, 0], mode="L")
                        elif c == 3:
                            pil_img = Image.fromarray(arr[:, :, :3], mode="RGB")
                        elif c >= 4:
                            pil_img = Image.fromarray(arr[:, :, :4], mode="RGBA")
                        else:
                            return None
                    else:
                        return None
                    
                    pil_img.save(os.path.join(tmpdir, name), compress_level=1)
                    return {"filename": name, "subfolder": "", "type": "temp"}
                except Exception as e:
                    print(f"[IPM] Warning: Failed to save {tag} image: {e}")
                    return None
            
            # Try to save all three images
            result_save = _save(result_u8[:, :, :3], "r")
            bg_save = _save(bg_np, "b")
            layer_save = _save(layer_np, "l")
            
            # Build ui_images, filtering out None values
            ui_images = [x for x in [result_save, bg_save, layer_save] if x is not None]
            
        except Exception as e:
            print(f"[IPM] Error saving preview images: {e}")
            ui_images = []

        return {
            "ui": {"images": ui_images},
            "result": (result_tensor, warped_layer_tensor, result_mask_tensor, corners_out_str),
        }


class Example:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("Image", { "tooltip": "This is an image"}),
                "int_field": ("INT", {
                    "default": 0,
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "float_field": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number"}),
                "print_to_screen": (["enable", "disable"],),
                "string_field": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Hello World!"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "test"

    #OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Example"

    def test(self, image, string_field, int_field, float_field, print_to_screen):
        if print_to_screen == "enable":
            print(f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """)
        #do some processing on the image, in this example I just invert it
        image = 1.0 - image
        return (image,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


class LoadImageFromURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "timeout_seconds": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load"
    CATEGORY = "ComfyUI_tools_for_longpean_zsy"

    def load(self, url: str, timeout_seconds: int):
        request = urllib.request.Request(
            url=url.strip(),
            headers={
                "User-Agent": "Mozilla/5.0 (ComfyUI URL Image Loader)",
                "Accept": "image/*,*/*;q=0.8",
            },
        )

        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            image_bytes = response.read()

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        image_np = np.array(pil_image).astype(np.float32) / 255.0

        rgb = image_np[:, :, :3]
        alpha = image_np[:, :, 3]

        # Ensure proper tensor format: (batch, height, width, channels) with float32
        image_tensor = torch.from_numpy(rgb).unsqueeze(0).to(dtype=torch.float32)
        mask_tensor = torch.from_numpy(alpha).unsqueeze(0).to(dtype=torch.float32)
        
        # Ensure tensors are on CPU (ComfyUI standard)
        image_tensor = image_tensor.cpu()
        mask_tensor = mask_tensor.cpu()
        
        return (image_tensor, mask_tensor)


class FillRGBAAlphaZeroBackground:
    """
    Fill transparent pixels (alpha == 0) in an RGBA image with a clean HEX color.

    Output is RGB only, keeping the same resolution as the input image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "输入 RGBA 图像；若图像不含内嵌 Alpha，可额外连接 alpha_mask。",
                }),
                "fill_hex": ("STRING", {
                    "default": "#ffffff",
                    "multiline": False,
                    "tooltip": "填充颜色（HEX），例如 #ffffff / #000000 / #ff8800。",
                }),
                "invert_alpha": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否反转 Alpha（开启后：原本不透明区域视为透明区域）。",
                }),
            },
            "optional": {
                "alpha_mask": ("MASK", {
                    "tooltip": "可选 Alpha 通道（推荐连接 LoadImage 的 MASK 输出）。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "fill"
    CATEGORY = "ComfyUI_tools_for_longpean_zsy"

    @staticmethod
    def _parse_hex_color(fill_hex: str) -> tuple[float, float, float]:
        s = (fill_hex or "").strip().lower()
        if s.startswith("#"):
            s = s[1:]

        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)

        if len(s) != 6 or any(ch not in "0123456789abcdef" for ch in s):
            s = "ffffff"

        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return (r, g, b)

    def fill(
        self,
        image: torch.Tensor,
        fill_hex: str,
        invert_alpha: bool = False,
        alpha_mask: torch.Tensor = None,
    ):
        rgb = image[..., :3]

        if image.shape[-1] >= 4:
            alpha = image[..., 3]
        elif alpha_mask is not None:
            alpha = alpha_mask
        else:
            alpha = torch.ones_like(rgb[..., 0])

        if invert_alpha:
            alpha = 1.0 - alpha

        fill_color = torch.tensor(
            self._parse_hex_color(fill_hex),
            dtype=rgb.dtype,
            device=rgb.device,
        ).view(1, 1, 1, 3)

        transparent = (alpha <= 0.0).unsqueeze(-1)
        result_rgb = torch.where(transparent, fill_color, rgb)

        return (result_rgb,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "InteractivePerspectiveMixer": InteractivePerspectiveMixer,
    "Example": Example,
    "LoadImageFromURL": LoadImageFromURL,
    "FillRGBAAlphaZeroBackground": FillRGBAAlphaZeroBackground,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "InteractivePerspectiveMixer": "Perspective transformer Layer mixer",
    "Example": "Example Node",
    "LoadImageFromURL": "Load Image From URL",
    "FillRGBAAlphaZeroBackground": "Fill RGBA Transparent Pixels",
}
