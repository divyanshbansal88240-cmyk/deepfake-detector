"""Phase 1 - invisible watermarking & authentication.

Two complementary layers are embedded in every protected image:

1. **Robust layer (DCT-QIM).** A 64-bit payload (32-bit media ID + 32-bit
   HMAC-SHA256 tag) is embedded in mid-frequency 8x8 block-DCT coefficients of
   the luma channel with *dithered quantisation index modulation*.  The image is
   analysed at a fixed canonical size so the mark survives resizing, and each bit
   is spread over dozens of keyed pseudo-random blocks (soft-vote decoding) so it
   survives JPEG / H.264 re-compression, blur and noise.  The HMAC tag makes the
   mark unforgeable without the secret key.  Per-block vote agreement also gives
   a *tamper localisation map* (regions pasted in / re-synthesised by a
   face-swap do not agree with the surrounding watermark).

2. **Fragile layer (keyed LSB).** A keyed pseudo-random pattern in the LSB plane
   of the blue channel.  It survives only lossless storage (PNG) and gives
   pixel-exact integrity checking + tamper localisation in that case.  Once the
   file is laundered it is (by design) destroyed, which is itself informative.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage
from scipy.fft import dctn, idctn

BLOCK = 8


# --------------------------------------------------------------------- helpers
def _seed(key: bytes, label: bytes) -> int:
    return int.from_bytes(hashlib.sha256(key + b"|" + label).digest()[:8], "big")


def _bits(value: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(value, dtype=np.uint8)).astype(np.uint8)


def make_payload(media_id: int, key: bytes) -> np.ndarray:
    """64-bit payload = 32-bit ID || first 32 bits of HMAC-SHA256(key, ID)."""
    idb = int(media_id & 0xFFFFFFFF).to_bytes(4, "big")
    tag = hmac.new(key, idb, hashlib.sha256).digest()[:4]
    return np.concatenate([_bits(idb), _bits(tag)])


def expected_tag_bits(id_bits: np.ndarray, key: bytes) -> np.ndarray:
    idb = np.packbits(id_bits.astype(np.uint8)).tobytes()
    return _bits(hmac.new(key, idb, hashlib.sha256).digest()[:4])


def bits_to_int(bits: np.ndarray) -> int:
    return int.from_bytes(np.packbits(bits.astype(np.uint8)).tobytes(), "big")


# robustness <-> invisibility trade-off (PSNR on mixed test images in comments)
PROFILES = {
    "stealth":  dict(canonical_size=128, delta=40.0, coeffs=((1, 2), (2, 1), (2, 2))),   # ~40 dB
    "balanced": dict(canonical_size=128, delta=60.0, coeffs=((0, 2), (2, 0), (1, 1))),   # ~37 dB
    "robust":   dict(canonical_size=128, delta=80.0, coeffs=((0, 1), (1, 0), (1, 1))),   # ~34 dB
}


@dataclass
class WatermarkParams:
    canonical_size: int = 128
    delta: float = 60.0
    nbits: int = 64
    verify_threshold: float = 0.875
    suspicious_threshold: float = 0.68
    tamper_min_blocks: int = 4
    tamper_min_fraction: float = 0.03
    tamper_alpha: float = 0.4
    coeffs: tuple = ((0, 2), (2, 0), (1, 1))

    @classmethod
    def from_config(cls, cfg: dict) -> "WatermarkParams":
        w, t = cfg["watermark"], cfg["tamper"]
        base = dict(PROFILES[w.get("profile", "balanced")])
        return cls(nbits=w["nbits_id"] + w["nbits_tag"], verify_threshold=w["verify_threshold"],
                   suspicious_threshold=w["suspicious_threshold"], tamper_min_blocks=t["min_blocks"],
                   tamper_min_fraction=t["min_fraction"], tamper_alpha=t.get("alpha", 0.4), **base)

    @classmethod
    def from_profile(cls, name: str = "balanced", **kw) -> "WatermarkParams":
        return cls(**{**PROFILES[name], **kw})


@dataclass
class VerifyResult:
    verified: bool
    suspicious: bool            # watermark traces present but not verifiable
    tag_score: float            # fraction of HMAC tag bits that match (0.5 = chance)
    media_id: int
    bit_confidence: float       # mean |soft vote| in [0,1]
    block_agreement: float      # median per-block agreement in [0,1]
    tamper_map: np.ndarray      # (nb, nb) float, 1 = disagrees with watermark
    tamper_mask: np.ndarray     # (nb, nb) bool, connected flagged regions
    tamper_fraction: float
    tampered: bool
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------- core engine
class DCTWatermarker:
    def __init__(self, key: bytes | str, params: WatermarkParams | None = None):
        self.key = key.encode() if isinstance(key, str) else key
        self.p = params or WatermarkParams()
        n = self.p.canonical_size // BLOCK
        self.nb = n
        nblocks = n * n
        rng = np.random.default_rng(_seed(self.key, b"assign"))
        order = rng.permutation(nblocks)
        bit_of_block = np.empty(nblocks, dtype=np.int64)
        bit_of_block[order] = np.arange(nblocks) % self.p.nbits
        self.bit_of_block = bit_of_block.reshape(n, n)
        drng = np.random.default_rng(_seed(self.key, b"dither"))
        self.dither = drng.uniform(0, self.p.delta, size=(n, n, len(self.p.coeffs)))

    # -- canonical luma ------------------------------------------------------
    def _canon_luma(self, bgr: np.ndarray) -> np.ndarray:
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y = ycc[..., 0].astype(np.float32)
        s = self.p.canonical_size
        interp = cv2.INTER_AREA if min(y.shape) >= s else cv2.INTER_CUBIC
        return cv2.resize(y, (s, s), interpolation=interp)

    def _blocks(self, y: np.ndarray) -> np.ndarray:
        n = self.nb
        b = y.reshape(n, BLOCK, n, BLOCK).transpose(0, 2, 1, 3)
        return dctn(b, type=2, norm="ortho", axes=(2, 3))

    def _unblocks(self, d: np.ndarray) -> np.ndarray:
        b = idctn(d, type=2, norm="ortho", axes=(2, 3))
        return b.transpose(0, 2, 1, 3).reshape(self.p.canonical_size, self.p.canonical_size)

    # -- embed ---------------------------------------------------------------
    def embed(self, bgr: np.ndarray, media_id: int, lsb: bool = True) -> np.ndarray:
        if min(bgr.shape[:2]) < 64:
            raise ValueError("image too small to watermark (min side 64 px)")
        payload = make_payload(media_id, self.key)
        y_small = self._canon_luma(bgr)
        d = self._blocks(y_small)
        bits = payload[self.bit_of_block]                       # (n,n)
        delta = self.p.delta
        for k, (u, v) in enumerate(self.p.coeffs):
            off = bits * (delta / 2.0) + self.dither[..., k]
            c = d[..., u, v]
            d[..., u, v] = np.round((c - off) / delta) * delta + off
        y_wm = self._unblocks(d)
        residual = y_wm - y_small
        h, w = bgr.shape[:2]
        s = self.p.canonical_size
        interp = cv2.INTER_CUBIC if (h, w) != (s, s) else cv2.INTER_LINEAR
        residual_full = cv2.resize(residual, (w, h), interpolation=interp)
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ycc[..., 0] += residual_full
        out = cv2.cvtColor(np.clip(np.round(ycc), 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        if lsb:
            out = self.lsb_embed(out)
        return out

    # -- extract / verify ----------------------------------------------------
    def _soft(self, bgr: np.ndarray) -> np.ndarray:
        """Per-block, per-coefficient soft decision in [-1, 1]; >0 means bit 1."""
        d = self._blocks(self._canon_luma(bgr))
        delta = self.p.delta
        soft = np.empty(d.shape[:2] + (len(self.p.coeffs),), dtype=np.float64)
        for k, (u, v) in enumerate(self.p.coeffs):
            c = d[..., u, v] - self.dither[..., k]
            e0 = np.abs(c - np.round(c / delta) * delta)
            c1 = c - delta / 2.0
            e1 = np.abs(c1 - np.round(c1 / delta) * delta)
            soft[..., k] = (e0 - e1) / (delta / 2.0)             # e0+e1 = delta/2
        return np.clip(soft, -1, 1)

    def verify(self, bgr: np.ndarray) -> VerifyResult:
        p = self.p
        soft = self._soft(bgr)
        votes = np.zeros(p.nbits)
        np.add.at(votes, self.bit_of_block.ravel(), soft.sum(-1).ravel())
        counts = np.bincount(self.bit_of_block.ravel(), minlength=p.nbits) * soft.shape[-1]
        norm_votes = votes / counts
        bits = (norm_votes > 0).astype(np.uint8)
        id_bits, tag_bits = bits[:32], bits[32:]
        tag_score = float((tag_bits == expected_tag_bits(id_bits, self.key)).mean())
        verified = tag_score >= p.verify_threshold
        suspicious = (not verified) and tag_score >= p.suspicious_threshold

        # per-block agreement with the decoded bits -> tamper localisation
        decoded = bits[self.bit_of_block]                        # (n,n)
        hard = (soft > 0).astype(np.uint8)
        agree = (hard == decoded[..., None]).mean(-1)            # (n,n) in {0,1/3,2/3,1}
        agree_s = ndimage.uniform_filter(agree, size=3, mode="nearest")
        med = float(np.median(agree))
        tamper_map = 1.0 - agree_s
        mask = np.zeros_like(agree_s, dtype=bool)
        if verified and med >= 0.6:
            thr = 0.5 + p.tamper_alpha * (med - 0.5)
            raw = agree_s < thr
            lab, n = ndimage.label(raw)
            for i in range(1, n + 1):
                comp = lab == i
                if comp.sum() >= p.tamper_min_blocks:
                    mask |= comp
        frac = float(mask.mean())
        tampered = bool(verified and frac >= p.tamper_min_fraction)
        return VerifyResult(
            verified=bool(verified), suspicious=bool(suspicious), tag_score=tag_score,
            media_id=bits_to_int(id_bits), bit_confidence=float(np.abs(norm_votes).mean()),
            block_agreement=med, tamper_map=tamper_map, tamper_mask=mask,
            tamper_fraction=frac, tampered=tampered,
            extra={"lsb": self.lsb_check(bgr), "votes": norm_votes},
        )

    def verify_votes(self, votes: np.ndarray) -> tuple[bool, float, int]:
        """Decide from soft votes pooled over several frames -> (verified, tag_score, media_id)."""
        bits = (votes > 0).astype(np.uint8)
        score = float((bits[32:] == expected_tag_bits(bits[:32], self.key)).mean())
        return score >= self.p.verify_threshold, score, bits_to_int(bits[:32])

    # -- fragile LSB layer ---------------------------------------------------
    def _lsb_pattern(self, shape) -> np.ndarray:
        rng = np.random.default_rng(_seed(self.key, b"lsb%d_%d" % shape[:2]))
        return rng.integers(0, 2, size=shape[:2], dtype=np.uint8)

    def lsb_embed(self, bgr: np.ndarray) -> np.ndarray:
        out = bgr.copy()
        out[..., 0] = (out[..., 0] & 0xFE) | self._lsb_pattern(bgr.shape)
        return out

    def lsb_check(self, bgr: np.ndarray, block: int = 16) -> dict:
        """Pixel-exact integrity check.  state: intact | localised | absent."""
        pat = self._lsb_pattern(bgr.shape)
        mism = ((bgr[..., 0] & 1) != pat).astype(np.float32)
        h, w = mism.shape
        hb, wb = h // block, w // block
        if hb == 0 or wb == 0:
            return {"state": "absent", "match": float(1 - mism.mean()), "bad_fraction": 1.0}
        grid = mism[: hb * block, : wb * block].reshape(hb, block, wb, block).mean((1, 3))
        bad = grid > 0.25
        bad_frac = float(bad.mean())
        match = float(1 - mism.mean())
        if match > 0.999:
            state = "intact"
        elif bad_frac < 0.5 and match > 0.8:
            state = "localised"
        else:
            state = "absent"            # lossy re-encode / not our file
        return {"state": state, "match": match, "bad_fraction": bad_frac, "bad_map": bad}


# ---------------------------------------------------------------- quality metrics
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))
