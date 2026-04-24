import torch


class TissuePaperAlphaModifier:
    """
    Generate irregular alpha from an AO-like input and tint it with a HEX color.
    Outputs:
      - ao_with_alpha: original RGB + computed alpha (RGBA tensor)
      - tissue_with_color: multiply-colored RGB + computed alpha (RGBA tensor)
      - alpha_mask: computed alpha as MASK tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input AO-like image tensor, shape [B, H, W, C].",
                }),
                "hex_color": ("STRING", {
                    "default": "#ff0000",
                    "multiline": False,
                    "tooltip": "Hex color used to tint the tissue paper (e.g. #ff5555).",
                }),
                "min_alpha": ("FLOAT", {
                    "default": 0.60,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum alpha for brightest AO areas after mapping.",
                }),
                "max_alpha": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Maximum alpha for darkest AO areas after mapping.",
                }),
                "gamma": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Gamma applied to the normalized AO before alpha mapping.",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional alpha/mask input (e.g., LoadImage MASK). Overrides embedded alpha.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("ao_with_alpha", "tissue_with_color", "alpha_mask")
    FUNCTION = "apply"
    CATEGORY = "ComfyUI_tools_for_longpean_zsy"
    DESCRIPTION = "Compute irregular alpha from AO brightness and apply tint via multiply."

    @staticmethod
    def _parse_hex_color(hex_color: str, device, dtype) -> torch.Tensor:
        """Parse '#rrggbb' string to a 3-element tensor in [0,1]."""
        fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        if not isinstance(hex_color, str):
            return fallback
        s = hex_color.strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) != 6:
            return fallback
        try:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return torch.tensor([r, g, b], device=device, dtype=dtype) / 255.0
        except Exception:
            return fallback

    @staticmethod
    def _auto_contrast_per_image(ao_slice: torch.Tensor, alpha_slice: torch.Tensor) -> torch.Tensor:
        """
        Auto-contrast AO per image using only pixels where original alpha > 0.
        ao_slice: (H, W), alpha_slice: (H, W)
        """
        valid_mask = alpha_slice > 0
        if torch.any(valid_mask):
            valid_vals = ao_slice[valid_mask]
            ao_min = valid_vals.min()
            ao_max = valid_vals.max()
            if (ao_max - ao_min) > 1e-6:
                ao_norm = (ao_slice - ao_min) / (ao_max - ao_min)
                return torch.clamp(ao_norm, 0.0, 1.0)
        return torch.clamp(ao_slice, 0.0, 1.0)

    def apply(self, image: torch.Tensor, hex_color: str, min_alpha: float = 0.6,
              max_alpha: float = 1.0, gamma: float = 4.0, mask: torch.Tensor = None):
        # Clamp input to a sane range and unpack channels
        image = torch.clamp(image, 0.0, 1.0)
        b, h, w, c = image.shape
        rgb = image[..., :3]
        if mask is not None:
            # MASK shape: [B, H, W]
            original_alpha = torch.clamp(mask, 0.0, 1.0)
        elif c >= 4:
            original_alpha = image[..., 3]
        else:
            original_alpha = torch.ones((b, h, w), device=image.device, dtype=image.dtype)

        # Grayscale AO
        ao = (
            rgb[..., 0] * 0.299 +
            rgb[..., 1] * 0.587 +
            rgb[..., 2] * 0.114
        )

        # Auto-contrast per batch sample within alpha > 0 area
        ao_normalized = torch.empty_like(ao)
        for i in range(b):
            ao_normalized[i] = self._auto_contrast_per_image(ao[i], original_alpha[i])

        # Gamma correction
        ao_gamma = torch.pow(torch.clamp(ao_normalized, 0.0, 1.0), gamma)

        # Alpha mapping
        span = max_alpha - min_alpha
        calculated_alpha = max_alpha - ao_gamma * span
        calculated_alpha = torch.clamp(calculated_alpha, 0.0, 1.0)

        # Preserve original transparency strictly
        final_alpha = torch.clamp(calculated_alpha * original_alpha, 0.0, 1.0)

        # Parse color and apply multiply tint
        color_tensor = self._parse_hex_color(hex_color, image.device, image.dtype)
        color_broadcast = color_tensor.view(1, 1, 1, 3)
        color_rgb = torch.clamp(rgb * color_broadcast, 0.0, 1.0)

        ao_with_alpha = torch.cat([rgb, final_alpha.unsqueeze(-1)], dim=-1)
        tissue_with_color = torch.cat([color_rgb, final_alpha.unsqueeze(-1)], dim=-1)
        alpha_mask = final_alpha

        return (ao_with_alpha, tissue_with_color, alpha_mask)


NODE_CLASS_MAPPINGS = {
    "TissuePaperAlphaModifier": TissuePaperAlphaModifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TissuePaperAlphaModifier": "Tissue Paper Alpha Modifier",
}
