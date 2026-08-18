import torch

from model.model import VCC


def main():
    model = VCC(
        in_chans=3,
        backbone_name="convnext_tiny",
        load_pretrained=False,
        temporal_load_pretrained=False,
        save_wave_images=False,
    )

    removed_modules = [
        "two_frame_dwt",
        "low_align_net",
        "high_align_net",
        "temporal_dfgf_s4",
        "low_film_s4",
        "spatial_density_s4",
        "advanced_fusion_s4",
        "temporal_dfgf_s3",
        "low_film_s3",
        "spatial_density_s3",
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

    print("Spatial Wavelet ablation check passed.")


if __name__ == "__main__":
    main()
