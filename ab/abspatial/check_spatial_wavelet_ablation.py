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
        "freq_spatial_fuse_s4",
        "branch_concat_s4",
        "reduce_s4",
        "low_film_s3",
        "temporal_dfgf_s3",
        "freq_spatial_fuse_s3",
        "branch_concat_s3",
        "regression_head",
    ]
    for name in required_modules:
        if not hasattr(model, name):
            raise AssertionError(f"Expected module is missing: {name}")

    removed_modules = [
        "spatial_density_s4",
        "spatial_density_s3",
        "advanced_fusion_s4",
        "advanced_fusion_s3",
    ]
    for name in removed_modules:
        if hasattr(model, name):
            raise AssertionError(f"Unexpected module remains: {name}")

    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    if not isinstance(y, torch.Tensor):
        raise AssertionError(f"Expected tensor output, got {type(y)!r}")
    if tuple(y.shape) != (1, 16, 16):
        raise AssertionError(f"Expected output shape (1, 16, 16), got {tuple(y.shape)}")

    print("Spatial Wavelet + Temporal Frequency ablation check passed.")


if __name__ == "__main__":
    main()
