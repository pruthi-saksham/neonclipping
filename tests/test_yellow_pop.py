import shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import captions, layouts


def words(text, step=.45):
    return [{'start': .3 + i * step, 'end': .7 + i * step, 'text': w} for i, w in enumerate(text.split())]


def render(style='yellow_pop', anim='highlight', appearance=None, text='he died off of a foot shot who dies when'):
    with tempfile.TemporaryDirectory() as d:
        captions.write(words(text), style, Path(d), 'x', '', 538, 300, 1080, 1920, anim, appearance)
        return (Path(d) / 'x.ass').read_text(encoding='utf-8')


def events(ass):
    return [l for l in ass.split('\n') if l.startswith('Dialogue: 1')]


def plain(line):
    body = line.split(',,', 1)[1]
    return ''.join(part.split('}')[-1] for part in body.split('{') if part)


class YellowPop(unittest.TestCase):
    def test_montserrat_white_with_the_spoken_word_in_yellow(self):
        ass = render()
        self.assertIn('Style: Main,Montserrat ExtraBold,', ass)
        for line in events(ass):
            self.assertIn(r'\c&H000BD5FE&', line)          # one yellow (#FED50B) word per event
            self.assertIn(r'\fs80\bord6\3c&H00000000&', line)
            self.assertEqual(plain(line), plain(line).upper())

    def test_lines_hold_two_or_three_words(self):
        lines = {plain(l) for l in events(render())}
        self.assertTrue(lines)
        for line in lines:
            self.assertLessEqual(len(line.split()), 3, line)

    def test_only_the_first_word_of_a_line_pops_the_line_in(self):
        groups = captions.group_words(words('he died off of a foot shot who dies when'), max_words=3)
        evs = events(render())
        firsts = 0
        for line in evs:
            if r'\fscx90\fscy90\t(0,100,\fscx100\fscy100)' in line:
                firsts += 1
        self.assertEqual(firsts, len(groups))
        self.assertNotIn(r'\fscx118', ''.join(evs))         # no per-word bounce

    def test_a_chosen_font_and_group_size_still_win(self):
        ass = render(appearance={'caption_font': 'Arial Black', 'caption_max_words': 2})
        self.assertIn('Style: Main,Arial Black,', ass)
        for line in events(ass):
            self.assertLessEqual(len(plain(line).split()), 2)

    def test_other_styles_keep_arial_and_four_words(self):
        ass = render('hormozi', text='one two three four five six seven eight')
        self.assertIn('Style: Main,Arial,', ass)
        self.assertTrue(any(len(plain(l).split()) == 4 for l in events(ass)))


class Seam(unittest.TestCase):
    def test_stack_is_the_bottom_of_the_facecam(self):
        self.assertEqual(layouts.seam_y('stack', .32), 614)
        self.assertEqual(layouts.seam_y('stack', .4), 768)
        self.assertEqual(layouts.seam_y('bottom', .32), 1920 - 614)

    def test_blur_is_the_top_of_the_picture(self):
        self.assertEqual(layouts.seam_y('blur', .32, 1080, 1920, 16 / 9), 656)
        self.assertEqual(layouts.seam_y('blur', .32, 1080, 1920, 4 / 3), 555)   # 810 px tall picture

    def test_other_layouts_have_no_seam(self):
        self.assertIsNone(layouts.seam_y('fill'))


class Fonts(unittest.TestCase):
    def test_the_font_ships_with_its_licence_and_is_copied_for_ffmpeg(self):
        root = Path(captions.__file__).resolve().parent / 'fonts'
        self.assertTrue((root / 'Montserrat-ExtraBold.ttf').exists())
        self.assertIn('SIL OPEN FONT LICENSE', (root / 'OFL.txt').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as d:
            out = captions.font_dir(d)
            self.assertEqual(out, Path(d) / 'fonts')
            self.assertEqual((out / 'Montserrat-ExtraBold.ttf').stat().st_size, (root / 'Montserrat-ExtraBold.ttf').stat().st_size)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'needs ffmpeg')
    def test_libass_draws_it_from_the_shipped_font(self):
        """Rendered with only fonts/ (no system Montserrat needed): the yellow word really shows up in the frame."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            captions.write(words('do it'), 'yellow_pop', d, 'x', '', 538, 300, 1080, 1920, 'highlight', None)
            captions.font_dir(d)
            out = d / 'f.png'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=gray:s=1080x1920:d=1', '-vf', 'ass=x.ass:fontsdir=fonts',
                            '-ss', '0.5', '-frames:v', '1', str(out)], check=True, capture_output=True, cwd=d)
            raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(out), '-vf', 'crop=1080:160:0:458', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                                 capture_output=True, check=True).stdout
            yellow = sum(1 for i in range(0, len(raw), 3) if raw[i] > 230 and 190 < raw[i + 1] < 235 and raw[i + 2] < 60)
            self.assertGreater(yellow, 500)


if __name__ == '__main__':
    unittest.main()
