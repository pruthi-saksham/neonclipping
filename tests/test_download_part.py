import shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


def _make_source(path, seconds=120):
    """A source whose sound is unique at every moment (seeded pink noise), with a small picture."""
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'color=c=gray:s=64x36:r=10:d={seconds}',
                    '-f', 'lavfi', '-i', f'anoisesrc=c=pink:a=0.4:d={seconds}:seed=7', '-shortest',
                    '-c:v', 'libx264', '-g', '10', '-c:a', 'aac', str(path)], check=True, capture_output=True)


class FakeYDL:
    """Stands in for yt-dlp: cuts the asked range like yt-dlp does on a long HLS VOD, starting `early` seconds before the
    asked start and keeping the asked LENGTH from there."""
    calls = []

    def __init__(self, source, early):
        self.source, self.early = source, early

    def __call__(self, opts):
        self.opts = opts
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def download(self, urls):
        (r,) = list(self.opts['download_ranges']({'duration': 120}, None))
        a, z = r['start_time'], r['end_time']
        FakeYDL.calls.append((a, z))
        out = self.opts['outtmpl'].replace('%(ext)s', 'mp4')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{max(0, a - self.early)}', '-t', f'{z - a}', '-i', str(self.source),
                        '-c:v', 'libx264', '-c:a', 'aac', out], check=True, capture_output=True)


@unittest.skipUnless(shutil.which('ffmpeg'), 'needs ffmpeg')
class DownloadPartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.source = self.dir / 'source.mp4'
        _make_source(self.source)
        self.audio = self.dir / 'source.audio.m4a'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(self.source), '-vn', '-c:a', 'copy', str(self.audio)], check=True, capture_output=True)
        self.old = app.DATA, app._yt
        app.DATA = self.dir
        FakeYDL.calls = []

    def tearDown(self):
        app.DATA, app._yt = self.old
        self.tmp.cleanup()

    def test_a_part_that_starts_early_is_asked_again_longer(self):
        app._yt = FakeYDL(self.source, early=8)
        start, end = 50.0, 85.0                       # the clip
        a, z = start - app.PAD, end + 1
        self.assertAlmostEqual(app.part_shortfall(self.audio, self._cut(a - 8, z - a), a, end + .5), 7.5, delta=.15)
        name = app.download_part('url', 'best', a, z, self.audio, end + .5)
        part = self.dir / (name + '.mp4')
        begins = app.section_start(self.audio, part, a)
        self.assertAlmostEqual(begins, a - 8, delta=.1)
        self.assertGreaterEqual(begins + app._dur(part), end + .5 - .05)   # the clip's end is in the part now
        self.assertEqual(len(FakeYDL.calls), 2)
        self.assertEqual(sorted(p.name for p in self.dir.glob('*.mp4') if p.stem not in ('source', 'cut')), [part.name])   # the short part was removed

    def test_an_exact_part_is_downloaded_once(self):
        app._yt = FakeYDL(self.source, early=0)
        name = app.download_part('url', 'best', 46.0, 86.0, self.audio, 85.5)
        self.assertEqual(len(FakeYDL.calls), 1)
        self.assertTrue((self.dir / (name + '.mp4')).exists())

    def test_without_the_sound_track_nothing_is_measured(self):
        app._yt = FakeYDL(self.source, early=8)
        app.download_part('url', 'best', 46.0, 86.0)
        self.assertEqual(len(FakeYDL.calls), 1)

    def _cut(self, start, length):
        out = self.dir / 'cut.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{start}', '-t', f'{length}', '-i', str(self.source), '-c:v', 'libx264', '-c:a', 'aac', str(out)],
                       check=True, capture_output=True)
        return out


if __name__ == '__main__':
    unittest.main()
