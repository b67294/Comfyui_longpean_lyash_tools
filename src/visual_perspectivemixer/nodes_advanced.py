from inspect import cleandoc
import ast
import os
import uuid
import numpy as np
import cv2
import torch
from PIL import Image
import folder_paths


class InteractivePerspectiveMixerAdvanced:
    """
    Interactive Perspective Mixer (Advanced)

    Advanced rules focused on RGBA output consistency:
    1) If background_image is RGBA, result_image is RGBA.
    2) Result alpha channel policy:
       - If background_alpha is NOT connected: use background_image embedded alpha as-is.
       - If background_alpha is connected: use background_alpha directly.
    3) Before output, pixels where final alpha == 0 are forced to white in RGB.

    Corner order: TL -> TR -> BR -> BL (relative coordinates).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE", {
                    "tooltip": "Background image (RGB or RGBA).",
                }),
                "layer_image": ("IMAGE", {
                    "tooltip": "Layer/decal image to be perspective warped.",
                }),
                "blend_mode": (["multiply", "normal"], {
                    "default": "multiply",
                    "tooltip": "multiply: (BG * Layer)/255, normal: direct overlay in warped area.",
                }),
                "invert_bg_alpha": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert final background alpha source (optional behavior).",
                }),
            },
            "optional": {
                "background_alpha": ("MASK", {
                    "tooltip": "Optional alpha override for output alpha channel.",
                }),
                "layer_alpha": ("MASK", {
                    "tooltip": "Optional alpha for layer image.",
                }),
                "corners_input": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional corners: [(x,y),(x,y),(x,y),(x,y)] in TL,TR,BR,BL order.",
                }),
                "save_preview": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable to save 3 preview images (result+background+layer) for Open Editor; disable to save no previews (saves disk space)",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("result_image", "warped_layer", "result_mask", "corners_output")
    FUNCTION = "apply_perspective_advanced"
    CATEGORY = "ComfyUI_tools_for_longpean_zsy"
    DESCRIPTION = cleandoc(__doc__)

    def _default_corners(self, bg_w: int, bg_h: int, layer_w: int, layer_h: int) -> list[dict]:
        scale = min(bg_w / max(layer_w, 1), bg_h / max(layer_h, 1)) * 0.8
        w_rel = (layer_w * scale) / max(bg_w, 1)
        h_rel = (layer_h * scale) / max(bg_h, 1)
        cx, cy = 0.5, 0.5
        x0, y0 = cx - w_rel / 2, cy - h_rel / 2
        x1, y1 = cx + w_rel / 2, cy + h_rel / 2
        return [
            {"x": x0, "y": y0},  # TL
            {"x": x1, "y": y0},  # TR
            {"x": x1, "y": y1},  # BR
            {"x": x0, "y": y1},  # BL
        ]

    @staticmethod
    def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
        return (t[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _mask_tensor_to_uint8(m: torch.Tensor) -> np.ndarray:
        return (m[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _split_alpha(img: np.ndarray):
        if img.ndim == 3 and img.shape[2] >= 4:
            return img[:, :, :3], img[:, :, 3], True
        return img[:, :, :3], np.full(img.shape[:2], 255, dtype=np.uint8), False

    @staticmethod
    def _parse_corners_input(s: str):
        if not s or not s.strip():
            return None
        try:
            parsed = ast.literal_eval(s.strip())
            if (
                isinstance(parsed, (list, tuple))
                and len(parsed) == 4
                and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in parsed)
            ):
                return [{"x": float(x), "y": float(y)} for x, y in parsed]
        except Exception:
            return None
        return None

    @staticmethod
    def _corners_to_output_str(corners_list: list[dict]) -> str:
        pts = [(round(c["x"], 6), round(c["y"], 6)) for c in corners_list]
        return "[" + ",".join(f"({x},{y})" for x, y in pts) + "]"

    def apply_perspective_advanced(
        self,
        background_image: torch.Tensor,
        layer_image: torch.Tensor,
        blend_mode: str,
        invert_bg_alpha: bool = False,
        background_alpha: torch.Tensor = None,
        layer_alpha: torch.Tensor = None,
        corners_input: str = "",
        save_preview: bool = False,
    ):
        bg_np = self._tensor_to_uint8(background_image)
        layer_np = self._tensor_to_uint8(layer_image)

        bg_h, bg_w = bg_np.shape[:2]
        layer_h, layer_w = layer_np.shape[:2]

        bg_rgb, bg_alpha_embedded, bg_has_embedded_alpha = self._split_alpha(bg_np)
        layer_rgb, layer_alpha_embedded, layer_has_embedded_alpha = self._split_alpha(layer_np)

        # Final output alpha source policy
        if background_alpha is not None:
            out_alpha_u8 = self._mask_tensor_to_uint8(background_alpha)
        else:
            out_alpha_u8 = bg_alpha_embedded

        if invert_bg_alpha:
            out_alpha_u8 = 255 - out_alpha_u8

        # Layer alpha for compositing
        if layer_alpha is not None:
            layer_alpha_u8 = self._mask_tensor_to_uint8(layer_alpha)
        else:
            layer_alpha_u8 = layer_alpha_embedded

        active_corners = self._parse_corners_input(corners_input)
        if active_corners is None:
            active_corners = self._default_corners(bg_w, bg_h, layer_w, layer_h)

        dst_pts = np.array(
            [[c["x"] * bg_w, c["y"] * bg_h] for c in active_corners],
            dtype=np.float32,
        )

        src_pts = np.array(
            [
                [0, 0],
                [layer_w - 1, 0],
                [layer_w - 1, layer_h - 1],
                [0, layer_h - 1],
            ],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped_rgb = cv2.warpPerspective(
            layer_rgb,
            M,
            (bg_w, bg_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        warped_alpha_u8 = cv2.warpPerspective(
            layer_alpha_u8,
            M,
            (bg_w, bg_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        warped_alpha = warped_alpha_u8.astype(np.float32) / 255.0
        warped_alpha_3ch = warped_alpha[:, :, np.newaxis]

        bg_f = bg_rgb.astype(np.float32)
        layer_f = warped_rgb.astype(np.float32)

        if blend_mode == "multiply":
            blended = (bg_f * layer_f) / 255.0
        else:
            blended = layer_f

        result_rgb_f = bg_f * (1.0 - warped_alpha_3ch) + blended * warped_alpha_3ch
        result_rgb_u8 = result_rgb_f.clip(0, 255).astype(np.uint8)

        # Transparent pixels become white in RGB before output
        transparent_mask = out_alpha_u8 == 0
        if np.any(transparent_mask):
            result_rgb_u8[transparent_mask] = 255

        should_output_rgba = bg_has_embedded_alpha or (background_alpha is not None)

        if should_output_rgba:
            result_rgba_u8 = np.concatenate([result_rgb_u8, out_alpha_u8[:, :, np.newaxis]], axis=-1)
            result_tensor = torch.from_numpy(result_rgba_u8.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            result_tensor = torch.from_numpy(result_rgb_u8.astype(np.float32) / 255.0).unsqueeze(0)

        original_layer_has_alpha = (layer_alpha is not None) or layer_has_embedded_alpha
        if original_layer_has_alpha:
            warped_layer_u8 = np.concatenate([warped_rgb, warped_alpha_u8[:, :, np.newaxis]], axis=-1)
        else:
            warped_layer_u8 = warped_rgb
        warped_layer_tensor = torch.from_numpy(warped_layer_u8.astype(np.float32) / 255.0).unsqueeze(0)

        result_mask_tensor = torch.from_numpy(out_alpha_u8.astype(np.float32) / 255.0).unsqueeze(0)
        corners_out_str = self._corners_to_output_str(active_corners)

        ui_images = []
        
        if save_preview:
            try:
                tmpdir = folder_paths.get_temp_directory()
                uid = uuid.uuid4().hex[:10]

                def _save(arr, tag):
                    if arr is None or getattr(arr, "size", 0) == 0:
                        return None

                    name = f"ipm_adv_{tag}_{uid}.png"
                    try:
                        if arr.dtype != np.uint8:
                            if arr.dtype in [np.float32, np.float64]:
                                arr = (arr * 255).clip(0, 255).astype(np.uint8)
                            else:
                                arr = arr.astype(np.uint8)

                        if arr.ndim == 2:
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
                        print(f"[IPM-ADV] Warning: failed to save {tag}: {e}")
                        return None

                # Keep index convention aligned with frontend: [result, background, layer]
                result_preview = _save(result_rgb_u8, "r")
                background_preview = _save(bg_np, "b")
                layer_preview = _save(layer_np, "l")

                # Fallback to preserve 3 preview entries when one save fails.
                if result_preview is None:
                    result_preview = background_preview or layer_preview
                if background_preview is None:
                    background_preview = result_preview or layer_preview
                if layer_preview is None:
                    layer_preview = result_preview or background_preview

                ui_images = [x for x in [result_preview, background_preview, layer_preview] if x is not None]
            except Exception as e:
                print(f"[IPM-ADV] Error saving preview images: {e}")
                ui_images = []

        return {
            "ui": {"images": ui_images},
            "result": (result_tensor, warped_layer_tensor, result_mask_tensor, corners_out_str),
        }


NODE_CLASS_MAPPINGS = {
    "InteractivePerspectiveMixerAdvanced": InteractivePerspectiveMixerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InteractivePerspectiveMixerAdvanced": "Perspective Mixer Advanced (RGBA Alpha Policy)",
}
