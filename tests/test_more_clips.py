import json, os, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['NEONCLIPPING_NO_AUTOUPDATE'] = '1'
import curation


class ExcludeTests(unittest.TestCase):
    def test_overlap_is_a_share_of_the_shorter_span(self):
        self.assertAlmostEqual(curation.overlap({'start': 0, 'end': 10}, {'start': 5, 'end': 30}), .5)
        self.assertEqual(curation.overlap({'start': 0, 'end': 10}, {'start': 20, 'end': 30}), 0)

    def test_moments_that_already_have_clips_are_dropped(self):
        clips = [{'start': 0, 'end': 30}, {'start': 100, 'end': 140}, {'start': 205, 'end': 240}, {'start': 300, 'end': 330}]
        kept = curation.drop_excluded(clips, [(95, 135), (230, 260), (400, 420)])
        # 100-140 is 87% inside an earlier clip; 205-240 only touches one by 10 s of 35 (29%), so it stays
        self.assertEqual([c['start'] for c in kept], [0, 205, 300])
        self.assertEqual(curation.drop_excluded(clips, []), clips)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            import app as neon_app
        except Exception as error:      # noqa: BLE001 - optional in minimal environments
            raise unittest.SkipTest('TestClient unavailable: ' + str(error))
        cls.app = neon_app
        cls.client = TestClient(neon_app.app, client=('127.0.0.1', 50000))

    def test_malformed_exclude_is_refused_before_any_work_starts(self):
        for bad in ([{'start': 5}], [{'start': 10, 'end': 5}], 'x', [{'start': 'nan', 'end': 3}], [{'start': -1, 'end': 3}]):
            r = self.client.post('/make', json={'url': 'https://www.twitch.tv/videos/1', 'exclude': bad})
            self.assertEqual(r.status_code, 400, bad)
        self.assertGreaterEqual(self.app.MAX_CLIPS, 30)

    def test_audio_is_reused_only_from_a_finished_run_of_the_same_link(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(self.app, 'DATA', Path(d)):
            d = Path(d)
            (d / 'a1.state.json').write_text(json.dumps({'status': 'done', 'fast_path': True, 'source': {'url': 'U'}}))
            (d / 'a1.audio.mp4').write_bytes(b'x')
            (d / 'a1.audio.temp.mp4').write_bytes(b'x')
            (d / 'b2.state.json').write_text(json.dumps({'status': 'error', 'fast_path': True, 'source': {'url': 'V'}}))
            (d / 'b2.audio.mp4').write_bytes(b'x')
            (d / 'c3.state.json').write_text(json.dumps({'status': 'error', 'fast_path': True, 'source': {'url': 'X'}, 'meta': {'duration': 99.0}}))
            (d / 'c3.audio.mp4').write_bytes(b'x')
            self.assertEqual(self.app._cached_audio('U'), d / 'a1.audio.mp4')
            self.assertIsNone(self.app._cached_audio('V'))      # failed before its audio was in: may be incomplete
            self.assertEqual(self.app._cached_audio('X'), d / 'c3.audio.mp4')   # failed later: the audio is complete
            self.assertIsNone(self.app._cached_audio('W'))


class FitWordsTests(unittest.TestCase):
    def test_words_past_a_shortened_clip_are_trimmed(self):
        import subtitles
        words = [{'start': 1.0, 'end': 1.4, 'text': 'a'}, {'start': 9.8, 'end': 10.6, 'text': 'b'}, {'start': 10.2, 'end': 10.5, 'text': 'c'},
                 {'start': 'x', 'end': 1, 'text': 'bad'}]
        fitted = subtitles.fit_words(words, 10.0)
        self.assertEqual([(w['text'], w['end']) for w in fitted], [('a', 1.4), ('b', 10.0)])
        subtitles.validate_words(fitted, 10.0)        # what export checks: no longer refused
        with self.assertRaises(ValueError):
            subtitles.validate_words(words[:3], 10.0)  # the old behaviour that ended the whole batch


if __name__ == '__main__':
    unittest.main()
