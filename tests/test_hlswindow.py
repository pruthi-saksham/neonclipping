import json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hlswindow

PLAYLIST = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXT-X-PLAYLIST-TYPE:VOD
#EXTINF:10.000,
0.ts
#EXTINF:10.000,
1-muted.ts
#EXTINF:10.000,
2.ts
#EXTINF:4.500,
3.ts
#EXT-X-ENDLIST
"""


class PlaylistTests(unittest.TestCase):
    def test_parse_and_window(self):
        segs = hlswindow.parse(PLAYLIST, 'https://cdn.example/vod/chunked/index-dvr.m3u8')
        self.assertEqual([s[0] for s in segs], [0, 10, 20, 30])
        self.assertEqual(segs[1][2], 'https://cdn.example/vod/chunked/1-muted.ts')
        self.assertEqual(hlswindow.window(segs, 12, 25), segs[1:3])
        self.assertEqual(len(hlswindow.window(segs, 0, 10)), 1)        # a segment that only touches the end is not needed
        self.assertEqual(hlswindow.window(segs, 40, 50), [])

    def test_pick_needs_picture_and_sound(self):
        info = {'formats': [
            {'format_id': 'Audio_Only', 'protocol': 'm3u8_native', 'url': 'a', 'vcodec': 'none', 'acodec': 'mp4a', 'height': None},
            {'format_id': '160p', 'protocol': 'm3u8_native', 'url': 'b', 'vcodec': 'avc1', 'acodec': 'mp4a', 'height': 160, 'fps': 30},
            {'format_id': '720p60', 'protocol': 'm3u8_native', 'url': 'c', 'vcodec': 'avc1', 'acodec': 'mp4a', 'height': 720, 'fps': 60},
            {'format_id': '1080p60', 'protocol': 'm3u8_native', 'url': 'd', 'vcodec': 'avc1', 'acodec': 'mp4a', 'height': 1080, 'fps': 60},
            {'format_id': 'dash', 'protocol': 'https', 'url': 'e', 'vcodec': 'vp9', 'acodec': 'opus', 'height': 1080}]}
        self.assertEqual(hlswindow.pick(info, 1080)['format_id'], '1080p60')
        self.assertEqual(hlswindow.pick(info, 720)['format_id'], '720p60')
        self.assertEqual(hlswindow.pick(info, 240, lowest=True)['format_id'], '160p')
        self.assertIsNone(hlswindow.pick({'formats': info['formats'][:1]}, 1080))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'needs ffmpeg')
@unittest.skipIf(os.name == 'nt', 'file:// segment URLs are a POSIX-only test shortcut')
class WrapTests(unittest.TestCase):
    def test_window_after_the_26_5_hour_timestamp_wrap(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            # 30 s of video whose MPEG-TS timestamps start at 95430 s: they wrap (2^33 / 90 kHz = 95443.7 s) about 14 s in.
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=s=160x90:r=10:d=30', '-f', 'lavfi', '-i', 'sine=d=30',
                            '-output_ts_offset', '95430', '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '10', '-c:a', 'aac', '-shortest',
                            '-f', 'hls', '-hls_time', '2', '-hls_playlist_type', 'vod', '-hls_segment_filename', str(d / 'seg%d.ts'), str(d / 'index.m3u8')],
                           check=True, capture_output=True)
            out = d / 'window.mp4'
            begin = hlswindow.fetch((d / 'index.m3u8').as_uri(), 20, 25, out)
            self.assertAlmostEqual(begin, 20, delta=2)
            self.assertTrue(hlswindow.usable(out, 4))
            info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type:format=duration', '-of', 'json', str(out)],
                                             capture_output=True, text=True).stdout)
            self.assertEqual(sorted(s['codec_type'] for s in info['streams']), ['audio', 'video'])
            self.assertTrue(5 <= float(info['format']['duration']) <= 8)

    def test_usable_rejects_empty_files(self):
        with tempfile.TemporaryDirectory() as d:
            empty = Path(d) / 'empty.mp4'
            empty.write_bytes(b'')
            self.assertFalse(hlswindow.usable(empty))
            self.assertFalse(hlswindow.usable(Path(d) / 'missing.mp4'))


if __name__ == '__main__':
    unittest.main()
