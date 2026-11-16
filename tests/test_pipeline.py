import cv2
import numpy as np
import pytest
from skimage import data

from rg20 import laundering
from rg20.classical import ClassicalDetector
from rg20.pipeline import AUTHENTIC, DEEPFAKE, TAMPERED, UNVERIFIED, AuthenticationPipeline
from rg20.synthetic import make_fake, make_real
from rg20.watermark import DCTWatermarker


@pytest.fixture(scope="module")
def detector():
    rng = np.random.default_rng(0)
    reals = [make_real(rng) for _ in range(60)]
    fakes = [make_fake(make_real(rng), rng) for _ in range(60)]
    return ClassicalDetector().fit(reals + fakes, [0] * 60 + [1] * 60)


@pytest.fixture(scope="module")
def pipe(detector):
    return AuthenticationPipeline(DCTWatermarker("k"), detector)


def test_authentic_survives_laundering(pipe):
    img = cv2.resize(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (256, 256))
    marked = pipe.wm.embed(img, 11)
    assert pipe.analyze(marked).status == AUTHENTIC
    assert pipe.analyze(laundering.screen_record(marked, 1)).status in (AUTHENTIC, UNVERIFIED, TAMPERED)
    assert pipe.analyze(laundering.jpeg(marked, 2)).status == AUTHENTIC


def test_fake_without_watermark_flagged(pipe):
    rng = np.random.default_rng(123)
    fake = make_fake(make_real(rng), rng)
    r = pipe.analyze(fake)
    assert r.status in (DEEPFAKE, UNVERIFIED) and r.fake_prob is not None


def test_pasted_region_is_tampered(pipe):
    img = cv2.resize(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (512, 512))
    out = pipe.wm.embed(img, 2)
    out[100:240, 150:290] = cv2.resize(cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR), (140, 140))
    assert pipe.analyze(out).status == TAMPERED


def test_pipeline_without_detector():
    p = AuthenticationPipeline(DCTWatermarker("k"), None)
    r = p.analyze(np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8))
    assert r.status == UNVERIFIED and r.fake_prob is None


def test_laundering_ops_preserve_shape():
    x = np.random.default_rng(0).integers(0, 255, (100, 120, 3), dtype=np.uint8)
    for name in laundering.ATTACKS:
        for lv in (1, 2, 3):
            assert laundering.apply(x, name, lv).shape == x.shape
