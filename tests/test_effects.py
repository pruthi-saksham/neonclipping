import json, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import effects, hooks, layouts


def plan(slow=None, zooms=(), words=(), audio=True, opts=None, post=True, layout='blur'):
    opts = opts or {'sfx': True, 'zoom': True, 'slowmo': True, 'sfx_volume': 1.0}
    return effects.Plan(opts, 30.0, slow, list(zooms), list(words), 1080, 1920, '30', 30.0, audio, layout != 'blur', post,
                        effects.zoom_center(layout, .32, 1080, 1920))


class SettingsTests(unittest.TestCase):
    def test_off_by_default(self):
        self.assertEqual(effects.settings({}), {'sfx': False, 'zoom': False, 'slowmo': False, 'sfx_volume': 1.0})
        self.assertFalse(effects.wanted({}))
        self.assertTrue(effects.wanted({'zoom': True}))

    def test_validation(self):
        for body in ({'sfx': 'yes'}, {'zoom': 1}, {'sfx_volume': 2}, {'sfx_volume': 'x'}, {'sfx_volume': float('nan')}):
            with self.assertRaises(ValueError):
                effects.settings(body)
        with self.assertRaises(ValueError):
            hooks.settings({'slowmo': 'on'})
        for key in effects.KEYS:
            self.assertIn(key, hooks.KEYS)      # so Make clips, live clips, creators and saved-clip edits pass them through


class TimingTests(unittest.TestCase):
    def test_retime_stretches_only_the_slow_window(self):
        p = plan(slow=(10.0, 11.0))
        self.assertEqual(p.retime(5), 5)
        self.assertAlmostEqual(p.retime(10.5), 11.0)
        self.assertAlmostEqual(p.retime(11), 12.0)
        self.assertAlmostEqual(p.retime(20), 21.0)
        self.assertAlmostEqual(p.duration, 31.0)

    def test_caption_words_follow_slow_motion_and_saved_words_do_not_change(self):
        words = [{'start': 9.0, 'end': 9.4, 'text': 'before'}, {'start': 10.2, 'end': 10.6, 'text': 'inside'}, {'start': 12.0, 'end': 12.3, 'text': 'after'}]
        p = plan(slow=(10.0, 11.0), words=words)
        self.assertEqual([(w['start'], w['end']) for w in p.words], [(9.0, 9.4), (10.4, 11.2), (13.0, 13.3)])
        self.assertEqual(words[1]['start'], 10.2)

    def test_choose_keeps_effects_apart_and_inside_the_clip(self):
        score = np.zeros(int(40 / effects.HOP))
        for t, v in ((6, 2.0), (6.5, 1.9), (18, 2.5), (30, 3.0)):
            score[int(t / effects.HOP)] = v
        slow, zooms = effects.choose(score, 40.0, {'slowmo': True, 'zoom': True})
        self.assertTrue(2.0 <= slow[0] < slow[1] <= 40 - 1.4)
        self.assertAlmostEqual(slow[1] - slow[0], effects.SLOW_LEN)
        self.assertTrue(28.5 <= slow[0] <= 30)               # the strongest moment gets the slow motion
        self.assertTrue(zooms)
        for a, b in zooms:
            self.assertTrue(.2 <= a < b <= 40 - .4)
            self.assertTrue(b <= slow[0] - .3 or a >= slow[1])
        starts = [a for a, _ in zooms]
        self.assertTrue(all(y - x >= 3.0 for x, y in zip(starts, starts[1:])))

    def test_short_clips_get_no_slow_motion(self):
        slow, zooms = effects.choose(np.ones(100), 5.0, {'slowmo': True, 'zoom': True})
        self.assertIsNone(slow)


class GraphTests(unittest.TestCase):
    def test_slow_motion_replaces_the_source_input(self):
        p = plan(slow=(10.0, 11.0), zooms=[(3.0, 4.6)])
        graph, label = p.video('[0:v]scale=1080:1920[base]')
        self.assertEqual(label, '[fxbase]')
        self.assertEqual(graph.count('[0:v]'), 1)              # only the retime piece reads the source now
        self.assertIn('[fxsrc]scale=1080:1920[base]', graph)
        self.assertIn('concat=n=3:v=1:a=1[fxsrc][fxa]', graph)
        self.assertIn('perspective=', graph)
        self.assertIn('vignette=', graph)
        self.assertIn("eq=brightness=", graph)

    def test_templates_keep_the_card_still(self):
        p = plan(slow=(10.0, 11.0), zooms=[(3.0, 4.6)], post=False)
        graph, label = p.video('[0:v]scale=1080:1920[base]')
        self.assertEqual(label, '[base]')
        self.assertNotIn('perspective', graph)

    def test_audio_labels(self):
        p = plan(slow=(10.0, 11.0))
        self.assertEqual(p.audio(None, 1), ('', '[fxa]'))       # slow motion without effects: map the retimed voice
        piece, label = p.audio(None, 1, fade=True)
        self.assertEqual(label, '[fxfade]')
        self.assertIn('[fxa]afade', piece)
        self.assertEqual(p.voice('[0:a]aresample=48000[voice]'), '[fxa]aresample=48000[voice]')
        p.bed = Path('bed.wav')
        piece, label = p.audio('[aout]', 3)
        self.assertEqual(label, '[fxmix]')
        self.assertIn('[3:a]aformat', piece)
        self.assertIn('[aout]aformat', piece)
        silent = plan(audio=False)
        silent.bed = Path('bed.wav')
        self.assertEqual(silent.audio(None, 1), ('[1:a]aformat=sample_rates=48000:channel_layouts=stereo[fxbed];[fxbed]anull[fxmix]', '[fxmix]'))
        self.assertEqual(plan().audio('[aout]', 2), ('', '[aout]'))   # nothing to add: the music mix stays as it was

    def test_punch_ins_aim_at_the_face_on_stacked_layouts(self):
        cam = next(l for l in layouts.geometry('stack', .32, 1080, 1920)['layers'] if l['source'] == 'cam')
        self.assertEqual(effects.zoom_center('stack', .32, 1080, 1920), (cam['x'] + cam['w'] / 2, cam['y'] + cam['h'] / 2))
        self.assertEqual(effects.zoom_center('blur', .32, 1080, 1920), (540, 960))


class SoundTests(unittest.TestCase):
    def test_sounds_are_clean(self):
        for name in effects.SOUNDS:
            s = effects.sound(name)
            self.assertTrue(np.isfinite(s).all(), name)
            self.assertAlmostEqual(float(np.max(np.abs(s))), .89, places=2)
            self.assertTrue(.1 <= len(s) / effects.SR <= 2.0, name)

    def test_cues_and_bed(self):
        words = [{'start': 2.0 + i * .5, 'end': 2.3 + i * .5, 'text': t} for i, t in enumerate(['so', 'oh', 'my', 'god', 'run', 'run', 'ok', 'we', 'did', 'it'] * 4)]
        p = plan(slow=(20.0, 21.3), zooms=[(8.0, 9.6)], words=words)
        p.plan_sfx(words, np.zeros(int(30 / effects.HOP)))
        names = [n for n, _ in p.cues]
        for name in ('whoosh', 'hit', 'swell', 'boom'):
            self.assertIn(name, names)
        self.assertEqual([t for _, t in p.cues], sorted(t for _, t in p.cues))
        self.assertLessEqual(names.count('pop'), 3)
        boom = dict((n, t) for n, t in p.cues if n == 'boom')['boom']
        self.assertAlmostEqual(boom, 20.0)
        with tempfile.TemporaryDirectory() as d:
            bed = p.write_bed(Path(d) / 'bed.wav')
            import wave
            with wave.open(str(bed)) as f:
                self.assertEqual(f.getframerate(), effects.SR)
                self.assertGreaterEqual(f.getnframes() / effects.SR, p.duration)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'needs ffmpeg')
class RenderTests(unittest.TestCase):
    def test_render_keeps_picture_and_sound_together(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / 'src.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=s=320x180:r=30:d=12', '-f', 'lavfi', '-i',
                            "sine=f=300:d=12,volume='if(between(t,8,9),1,0.1)':eval=frame", '-c:v', 'libx264', '-preset', 'ultrafast',
                            '-c:a', 'aac', '-shortest', str(src)], check=True, capture_output=True)
            words = [{'start': 8.1, 'end': 8.5, 'text': 'oh'}, {'start': 8.5, 'end': 8.8, 'text': 'my'}, {'start': 8.8, 'end': 9.2, 'text': 'god'}]
            p = effects.build({'sfx': True, 'zoom': True, 'slowmo': True}, src, 0, 12, words, 180, 320, 'blur', d, 'job', top=.32, fps=30)
            self.assertIsNotNone(p.slow)
            graph, label = p.video(layouts.render_graph('blur', [(0, 0, 80, 60), (80, 0, 240, 180)], .32, 180, 320))
            piece, amap = p.audio(None, 1)
            out = d / 'out.mp4'
            r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), *p.inputs, '-filter_complex', graph + f';{label}null[v];' + piece,
                                '-map', '[v]', '-map', amap, '-t', str(p.duration), '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', str(out)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,duration', '-of', 'json', str(out)],
                                             capture_output=True, text=True).stdout)['streams']
            durations = {s['codec_type']: float(s['duration']) for s in info}
            self.assertAlmostEqual(durations['video'], p.duration, delta=.15)
            self.assertAlmostEqual(durations['audio'], p.duration, delta=.15)
            self.assertGreater(p.duration, 12.5)          # the slow-motion beat made the clip longer


if __name__ == '__main__':
    unittest.main()
