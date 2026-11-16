import shutil
import subprocess

import cv2
import numpy as np
import pytest
from skimage import data

from rg20.video import _write_video, extract_frames, verify_video, watermark_video
from rg20.watermark import DCTWatermarker

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def test_video_roundtrip_and_laundering(tmp_path):
    base = cv2.resize(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (320, 320))
    _write_video([np.roll(base, i * 2, 1) for i in range(30)], tmp_path / "o.mp4", 25, (320, 320), crf=18)
    wm = DCTWatermarker("k")
    watermark_video(tmp_path / "o.mp4", tmp_path / "w.mp4", wm, 77)
    r = verify_video(tmp_path / "w.mp4", wm)
    assert r["verified"] and r["media_id"] == 77
    assert not verify_video(tmp_path / "o.mp4", wm)["verified"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp_path / "w.mp4"), "-c:v", "libx264",
                    "-crf", "30", str(tmp_path / "l.mp4")], check=True)
    assert verify_video(tmp_path / "l.mp4", wm)["verified"]
    assert extract_frames(tmp_path / "o.mp4", tmp_path / "f", 30) >= 25
