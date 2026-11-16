"""Additional deterministic regression tests for the RG20 core.

These tests exercise public contracts with synthetic fixtures. They do not claim
model accuracy or replace evaluation on FaceForensics++ / DFDC.
"""
import cv2
import numpy as np
import pytest
from skimage import data

from rg20 import laundering
from rg20.classical import features
from rg20.faces import FaceExtractor
from rg20.metrics import classification_metrics
from rg20.pipeline import (
    AUTHENTIC,
    DEEPFAKE,
    TAMPERED,
    UNVERIFIED,
    AuthenticationPipeline,
)
from rg20.synthetic import make_fake, make_real
from rg20.viz import tamper_overlay
from rg20.watermark import (
    DCTWatermarker,
    WatermarkParams,
    bits_to_int,
    make_payload,
    psnr,
    ssim,
)


@pytest.fixture(scope="module")
def sample_image():
    return cv2.resize(
        cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR),
        (256, 256),
        interpolation=cv2.INTER_AREA,
    )


@pytest.mark.parametrize("media_id", [0, 1, 42, 2**31, 2**32 - 1])
def test_payload_roundtrips_unsigned_media_id(media_id):
    payload = make_payload(media_id, b"regression-key")
    assert payload.shape == (64,)
    assert set(np.unique(payload)).issubset({0, 1})
    assert bits_to_int(payload[:32]) == media_id


@pytest.mark.parametrize("media_id", [2, 17, 999, 2**32 - 1])
def test_watermark_embed_verify_roundtrip(sample_image, media_id):
    watermarker = DCTWatermarker("regression-key")
    marked = watermarker.embed(sample_image, media_id, lsb=False)
    result = watermarker.verify(marked)
    assert result.verified
    assert result.media_id == media_id
    assert result.tag_score >= watermarker.p.verify_threshold
    assert marked.shape == sample_image.shape
    assert marked.dtype == np.uint8


@pytest.mark.parametrize("profile", ["stealth", "balanced", "robust"])
def test_watermark_profiles_preserve_payload(sample_image, profile):
    watermarker = DCTWatermarker(
        "profile-key", WatermarkParams.from_profile(profile)
    )
    result = watermarker.verify(watermarker.embed(sample_image, 314, lsb=False))
    assert result.verified
    assert result.media_id == 314


@pytest.mark.parametrize("other_key", ["wrong", "another-key", ""])
def test_watermark_rejects_different_keys(sample_image, other_key):
    marked = DCTWatermarker("correct-key").embed(sample_image, 73, lsb=False)
    assert not DCTWatermarker(other_key).verify(marked).verified


@pytest.mark.parametrize(
    "width,height",
    [(64, 64), (96, 128), (127, 129), (256, 320), (384, 512)],
)
def test_watermark_supports_varied_image_geometry(width, height):
    image = cv2.resize(
        cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (width, height)
    )
    marked = DCTWatermarker("geometry-key").embed(image, 8, lsb=False)
    assert marked.shape == image.shape
    assert marked.dtype == np.uint8


@pytest.mark.parametrize(
    "attack",
    ["jpeg", "resize", "blur", "noise", "h264", "screen_record"],
)
@pytest.mark.parametrize("level", [1, 2, 3])
def test_laundering_attacks_preserve_image_contract(sample_image, attack, level):
    output = laundering.apply(sample_image, attack, level)
    assert output.shape == sample_image.shape
    assert output.dtype == np.uint8
    assert np.isfinite(output).all()
    assert output.min() >= 0 and output.max() <= 255


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_noise_attack_is_reproducible_for_fixed_seed(sample_image, seed):
    first = laundering.noise(sample_image, level=2, seed=seed)
    second = laundering.noise(sample_image, level=2, seed=seed)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("seed", [0, 7, 99, 1234, 2026])
def test_synthetic_real_generator_is_deterministic(seed):
    first = make_real(np.random.default_rng(seed), size=96)
    second = make_real(np.random.default_rng(seed), size=96)
    assert np.array_equal(first, second)
    assert first.shape == (96, 96, 3)
    assert first.dtype == np.uint8


@pytest.mark.parametrize("seed", [2, 5, 11, 23, 101])
def test_synthetic_fake_generator_preserves_shape_and_range(seed):
    rng = np.random.default_rng(seed)
    real = make_real(rng, size=96)
    fake = make_fake(real, rng)
    assert fake.shape == real.shape
    assert fake.dtype == np.uint8
    assert fake.min() >= 0 and fake.max() <= 255


@pytest.mark.parametrize(
    "height,width",
    [(32, 48), (64, 64), (96, 128), (224, 224), (320, 240)],
)
def test_full_frame_face_extractor_returns_fixed_size_crop(height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    extractor = FaceExtractor(backend="full", size=112)
    crop, box = extractor.crop(image)
    assert crop.shape == (112, 112, 3)
    assert crop.dtype == np.uint8
    assert box == (0, 0, width, height)


@pytest.mark.parametrize("threshold", [0.5, 0.6, 0.7, 0.8, 0.9])
def test_classification_metrics_are_bounded(threshold):
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    result = classification_metrics(labels, probabilities, threshold)
    for key in ("accuracy", "precision", "recall", "f1", "auc"):
        assert 0.0 <= result[key] <= 1.0
    assert result["n"] == 4


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.45])
def test_tamper_overlay_returns_valid_image(sample_image, alpha):
    tamper_map = np.zeros((16, 16), dtype=np.float32)
    tamper_map[5:9, 5:9] = 1.0
    mask = tamper_map > 0
    output = tamper_overlay(sample_image, tamper_map, mask, alpha=alpha)
    assert output.shape == sample_image.shape
    assert output.dtype == np.uint8


@pytest.mark.parametrize("fake_probability", [0.5, 0.7, 0.9])
def test_pipeline_marks_high_fake_probability_as_deepfake(sample_image, fake_probability):
    class FixedDetector:
        kind = "test"

        def predict_proba(self, images):
            assert len(images) == 1
            return np.array([fake_probability], dtype=float)

    pipeline = AuthenticationPipeline(
        DCTWatermarker("pipeline-key"), FixedDetector(), fake_threshold=0.5
    )
    report = pipeline.analyze(sample_image)
    assert report.status == DEEPFAKE
    assert report.fake_prob == pytest.approx(fake_probability)
    assert report.detector == "test"


def test_pipeline_without_detector_reports_unknown_provenance(sample_image):
    pipeline = AuthenticationPipeline(DCTWatermarker("pipeline-key"), None)
    report = pipeline.analyze(sample_image)
    assert report.status in (UNVERIFIED, TAMPERED)
    assert report.fake_prob is None


def test_pipeline_accepts_untampered_valid_watermark(sample_image):
    watermarker = DCTWatermarker("pipeline-key")
    marked = watermarker.embed(sample_image, 808)
    report = AuthenticationPipeline(watermarker).analyze(marked)
    assert report.status == AUTHENTIC
    assert report.wm.verified
    assert report.wm.media_id == 808


def test_classical_features_are_finite_and_one_dimensional(sample_image):
    vector = features(sample_image)
    assert vector.ndim == 1
    assert vector.size > 32
    assert np.isfinite(vector).all()


def test_image_quality_metrics_handle_identical_images(sample_image):
    assert psnr(sample_image, sample_image) == float("inf")
    assert ssim(sample_image, sample_image) == pytest.approx(1.0)


def test_watermark_rejects_too_small_image():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        DCTWatermarker("small-image-key").embed(image, 1)


def test_laundering_rejects_unknown_attack(sample_image):
    with pytest.raises(KeyError, match="unknown attack"):
        laundering.apply(sample_image, "not-an-attack", 1)
