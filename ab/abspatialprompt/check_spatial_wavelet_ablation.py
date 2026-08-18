import torch

from model.model import VCC


def main():
    model = VCC(
        in_chans=3,
        backbone_name="convnext_tiny",
        load_pretrained=False,
        temporal_load_pretrained=True,
        save_wave_images=False,
    )

    required_modules = [
        "two_frame_dwt",
        "wavelet_fusion",
        "backbone",
        "low_align_net",
        "high_align_net",
        "low_film_s4",
        "temporal_dfgf_s4",
        "spatial_density_s4",
        "branch_concat_s4",
        "reduce_s4",
        "low_film_s3",
        "temporal_dfgf_s3",
        "spatial_density_s3",
        "branch_concat_s3",
        "regression_head",
    ]
    for name in required_modules:
        if not hasattr(model, name):
            raise AssertionError(f"Expected module is missing: {name}")

    removed_modules = [
        "advanced_fusion_s4",
        "advanced_fusion_s3",
    ]
    for name in removed_modules:
        if hasattr(model, name):
            raise AssertionError(f"Unexpected module remains: {name}")

    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    if not isinstance(y, tuple) or len(y) != 2:
        raise AssertionError(f"Expected (density, aux) tuple output, got {type(y)!r}")
    density, aux = y
    density_s4, density_s3 = aux
    if tuple(density.shape) != (1, 16, 16):
        raise AssertionError(f"Expected output shape (1, 16, 16), got {tuple(density.shape)}")
    if tuple(density_s4.shape) != (1, 1, 4, 4):
        raise AssertionError(f"Expected S4 aux shape (1, 1, 4, 4), got {tuple(density_s4.shape)}")
    if tuple(density_s3.shape) != (1, 1, 8, 8):
        raise AssertionError(f"Expected S3 aux shape (1, 1, 8, 8), got {tuple(density_s3.shape)}")

    print("Spatial Prompt ablation check passed.")


if __name__ == "__main__":
    main()
