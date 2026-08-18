import torch

from model.model import VCC


def test_spatial_wavelet_ablation_forward_shape_and_modules():
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
        assert not hasattr(model, name)

    x = torch.randn(2, 3, 128, 128)
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 16, 16)
