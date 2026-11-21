import shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import highlights


def whole_file(source):
    """The previous implementation: everything decoded and analysed at once (memory grows with the length)."""
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source), '-vn', '-ac', '1', '-ar', '8000', '-f', 's16le', 'pipe:1']
    raw = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL).stdout
    x = np.frombuffer(raw[:len(raw) // 2 * 2], np.int16).astype(np.float32) / 32768
    n = len(x) // 8000
    if n == 0:
        return np.zeros(0), np.zeros(0)
    x = x[:n * 8000].reshape(n, 8000)
    spec = np.abs(np.fft.rfft(x * np.hanning(8000), axis=1)) ** 2
    return np.sqrt((x ** 2).mean(1)), spec[:, 2000:4000].sum(1) / (spec.sum(1) + 1e-9)


@unittest.skipUnless(shutil.which('ffmpeg'), 'needs ffmpeg')
class ChunkedAudioTests(unittest.TestCase):
    def test_chunks_give_the_same_signal_as_one_pass(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'a.m4a'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', "sine=f=3000:d=75.5,volume='0.2+0.7*gt(mod(t,9),6)':eval=frame",
                            '-f', 'lavfi', '-i', 'anoisesrc=c=pink:a=0.05:d=75.5', '-filter_complex', '[0][1]amix=inputs=2', '-c:a', 'aac', str(src)],
                           check=True, capture_output=True)
            old = highlights.AUDIO_CHUNK_SECONDS
            highlights.AUDIO_CHUNK_SECONDS = 20          # 75.5 s -> three full chunks and a short last one
            try:
                rms, hf = highlights._audio(src)
            finally:
                highlights.AUDIO_CHUNK_SECONDS = old
            ref_rms, ref_hf = whole_file(src)
            self.assertEqual(len(rms), 75)              # the trailing half second is dropped, as before
            np.testing.assert_allclose(rms, ref_rms, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(hf, ref_hf, rtol=1e-5, atol=1e-7)
            self.assertGreater(rms.max() / rms.min(), 2)  # the loud parts are really louder

    def test_rank_never_picks_skipped_spans_and_still_fills_the_count(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'b.m4a'
            # quiet pink noise with loud bursts at 40 s, 100 s, 160 s, 220 s and 280 s
            loud = '+'.join(f'between(t,{t},{t + 3})' for t in (40, 100, 160, 220, 280))
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f"anoisesrc=c=pink:a=0.5:d=330,volume='if({loud},1,0.03)':eval=frame",
                            '-c:a', 'aac', str(src)], check=True, capture_output=True)
            plain = highlights.rank(src, 330, count=3, length=20, motion_on=False)
            self.assertTrue(all(any(c['start'] <= t <= c['end'] for t in (40, 100, 160, 220, 280)) for c in plain))
            skip = [(90, 175)]                   # covers the bursts at 100 s and 160 s
            picked = highlights.rank(src, 330, count=3, length=20, motion_on=False, skip=skip)
            self.assertEqual(len(picked), 3)
            for c in picked:
                self.assertFalse(c['start'] < 175 and c['end'] > 90, c)
            self.assertEqual(sorted(any(c['start'] <= t <= c['end'] for c in picked) for t in (40, 220, 280)), [True, True, True])

    def test_no_sound_gives_empty_signals(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'v.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=2', '-c:v', 'libx264', str(src)],
                           check=True, capture_output=True)
            rms, hf = highlights._audio(src)
            self.assertEqual((len(rms), len(hf)), (0, 0))


if __name__ == '__main__':
    unittest.main()
