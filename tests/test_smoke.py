import os
import pytest

CKPT = os.environ.get("VISAEBENCH_SMOKE_CKPT")


@pytest.mark.skipif(
    not CKPT,
    reason="set VISAEBENCH_SMOKE_CKPT to a local SAE checkpoint dir or HF repo",
)
def test_evaluate_end_to_end_500_samples():
    import visaebench
    from visaebench.hub import load_sae

    sae = load_sae(CKPT, config_filename="config.yaml")
    results = visaebench.evaluate(
        sae=sae,
        backbone_name="clip_vitb16",
        max_samples=500,
    )
    assert len(results) == 7
    valid_dims = {
        "spatial_coherence",
        "reconstruction",
        "concept_detection",
        "disentanglement",
    }
    for r in results:
        assert r.dimension in valid_dims
        assert r.value == r.value, f"{r.name} returned NaN"
