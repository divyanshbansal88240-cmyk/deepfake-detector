import cv2
import numpy as np
import pytest
from skimage import data

from rg20 import laundering
from rg20.watermark import DCTWatermarker, WatermarkParams, psnr, make_payload


@pytest.fixture(scope="module")
def img():
    return cv2.resize(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (512, 512))


@pytest.fixture(scope="module")
def wm():
    return DCTWatermarker("unit-test-key")


def test_payload_is_64_bits_and_keyed():
    a, b = make_payload(42, b"k1"), make_payload(42, b"k2")
    assert a.shape == (64,) and (a[:32] == b[:32]).all() and (a[32:] != b[32:]).any()


def test_roundtrip_and_invisibility(img, wm):
    out = wm.embed(img, 123456)
    assert psnr(img, out) > 33
    r = wm.verify(out)
    assert r.verified and r.media_id == 123456 and r.tag_score == 1.0
    assert r.extra["lsb"]["state"] == "intact" and not r.tampered


def test_unmarked_and_wrong_key_rejected(img, wm):
    assert not wm.verify(img).verified
    assert not DCTWatermarker("another-key").verify(wm.embed(img, 1)).verified


@pytest.mark.parametrize("attack,level", [("jpeg", 2), ("resize", 2), ("blur", 1), ("noise", 2), ("h264", 1)])
def test_survives_laundering(img, wm, attack, level):
    out = laundering.apply(wm.embed(img, 99), attack, level)
    r = wm.verify(out)
    assert r.verified and r.media_id == 99
    assert r.extra["lsb"]["state"] == "absent"      # fragile layer is gone, robust layer remains


def test_paste_tamper_localised(img, wm):
    out = wm.embed(img, 7)
    other = cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR)
    out[100:220, 160:280] = cv2.resize(other, (120, 120))
    r = wm.verify(out)
    assert r.verified and r.tampered
    # flagged blocks overlap the pasted area (rows/cols in 8px-canonical blocks of a 128 grid)
    ys, xs = np.nonzero(r.tamper_mask)
    assert 0.1 < ys.mean() / r.tamper_mask.shape[0] < 0.6
    lsb = r.extra["lsb"]
    assert lsb["state"] == "localised"


def test_clean_image_not_flagged(img, wm):
    assert not wm.verify(laundering.jpeg(wm.embed(img, 7), 2)).tampered


def test_small_image_rejected(wm):
    with pytest.raises(ValueError):
        wm.embed(np.zeros((32, 32, 3), np.uint8), 1)


def test_non_square_and_odd_sizes(wm):
    im = cv2.resize(cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR), (701, 389))
    assert wm.verify(laundering.jpeg(wm.embed(im, 5), 1)).verified


def test_profiles():
    for p in ("stealth", "balanced", "robust"):
        w = DCTWatermarker("k", WatermarkParams.from_profile(p))
        im = cv2.resize(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (384, 384))
        assert w.verify(w.embed(im, 3)).verified
