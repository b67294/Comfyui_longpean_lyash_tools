#!/usr/bin/env python

"""Tests for `visual_perspectivemixer` package."""

import pytest
import torch

from src.visual_perspectivemixer.nodes import Example, FillRGBAAlphaZeroBackground

@pytest.fixture
def example_node():
    """Fixture to create an Example node instance."""
    return Example()

def test_example_node_initialization(example_node):
    """Test that the node can be instantiated."""
    assert isinstance(example_node, Example)

def test_return_types():
    """Test the node's metadata."""
    assert Example.RETURN_TYPES == ("IMAGE",)
    assert Example.FUNCTION == "test"
    assert Example.CATEGORY == "Example"


def test_fill_rgba_alpha_zero_background_fills_only_transparent_pixels():
    node = FillRGBAAlphaZeroBackground()

    # RGBA image (1, H, W, C)
    image = torch.tensor(
        [[
            [[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.6, 1.0]],
            [[0.9, 0.8, 0.7, 0.0], [0.2, 0.3, 0.4, 0.5]],
        ]],
        dtype=torch.float32,
    )

    (out,) = node.fill(image=image, fill_hex="#ff0000", invert_alpha=False)

    # alpha == 0 pixels -> filled red
    assert torch.allclose(out[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(out[0, 1, 0], torch.tensor([1.0, 0.0, 0.0]))

    # other pixels unchanged
    assert torch.allclose(out[0, 0, 1], torch.tensor([0.4, 0.5, 0.6]))
    assert torch.allclose(out[0, 1, 1], torch.tensor([0.2, 0.3, 0.4]))


def test_fill_rgba_alpha_zero_background_output_is_rgb_same_resolution():
    node = FillRGBAAlphaZeroBackground()
    image = torch.rand((1, 7, 9, 4), dtype=torch.float32)

    (out,) = node.fill(image=image, fill_hex="#ffffff", invert_alpha=False)

    assert tuple(out.shape) == (1, 7, 9, 3)


def test_fill_rgba_alpha_zero_background_supports_invert_alpha():
    node = FillRGBAAlphaZeroBackground()

    image = torch.tensor(
        [[
            [[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.6, 1.0]],
        ]],
        dtype=torch.float32,
    )

    (out,) = node.fill(image=image, fill_hex="#00ff00", invert_alpha=True)

    # invert 后：原 alpha=1 的像素会被视为透明并填充
    assert torch.allclose(out[0, 0, 1], torch.tensor([0.0, 1.0, 0.0]))

    # 原 alpha=0 的像素反转后不透明，保持原 RGB
    assert torch.allclose(out[0, 0, 0], torch.tensor([0.1, 0.2, 0.3]))
