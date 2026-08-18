import torch

from model.model import VCC


def test_spatial_wavelet_ablation_forward_shape_and_modules():
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
        assert hasattr(model, name)

    removed_modules = [
        "spatial_density_s4",
        "spatial_density_s3",
        "advanced_fusion_s4",
        "advanced_fusion_s3",
    ]
    for name in removed_modules:
        assert not hasattr(model, name)

    x = torch.randn(2, 3, 128, 128)
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 16, 16)
