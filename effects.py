"""Optional clip effects: sound effects, punch-in zooms and one slow-motion beat. Nothing changes unless a request asks for them.

  sfx=true           a whoosh on the opening, a hit on every punch-in, a swell into a boom on the slow-motion beat and a few soft
                     pops on shouted words. The sounds are synthesised here with numpy: no sample packs, nothing downloaded.
  zoom=true          1-4 quick punch-in zooms (ease in, hold, ease out) on the clip's loudest / most exclaimed moments.
  slowmo=true        the peak moment plays at half speed with a deeper voice, a white flash, a short camera shake, a vignette and
                     a slow push-in. Caption timings are moved to match, so they stay in sync.
  sfx_volume=0-1.5   loudness of the sound effects (default 1).

The moments come from the clip's own sound (a loudness envelope) plus its word-timed transcript (excite.word_weight). Zoom,
shake, flash and vignette work on the finished layout before the captions are burned in, so the captions stay sharp and never
zoom. app.export asks build() for a Plan and splices its pieces into the FFmpeg graph.
"""
import functools
import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np

KEYS = ('sfx', 'zoom', 'slowmo', 'sfx_volume')
SR = 48000          # sound-effect bed sample rate
HOP = 0.05          # seconds per loudness frame
SLOW = 0.5          # slow-motion speed
PITCH = 0.75        # voice pitch inside the slow-motion beat (lower = more dramatic)
SLOW_LEN = 1.3      # seconds of source played in slow motion
GAIN = {'whoosh': .5, 'swell': .45, 'boom': .9, 'hit': .7, 'pop': .35}
LAYERED_ZOOM = .8   # facecam/gameplay layouts zoom a little less: the whole stacked frame moves


def settings(body):
    """Validated effect settings (all off by default). Raises ValueError with a readable message."""
    out = {}
    for key in ('sfx', 'zoom', 'slowmo'):
        value = body.get(key, False)
        if value is None:
            value = False
        if not isinstance(value, bool):
            raise ValueError(f'{key} must be on or off.')
        out[key] = value
    raw = body.get('sfx_volume')
    try:
        volume = 1.0 if raw in (None, '') else float(raw)
    except (TypeError, ValueError):
        raise ValueError('Sound effect volume must be a number.')
    if not math.isfinite(volume) or not 0 <= volume <= 1.5:
        raise ValueError('Sound effect volume must be between 0 and 1.5.')
    out['sfx_volume'] = volume
    return out


def wanted(body):
    s = settings(body)
    return s['sfx'] or s['zoom'] or s['slowmo']


# ---------------------------------------------------------------- sound effects (synthesised, cached per process)
def _shaped_noise(n, centre, width, seed):
    """Noise whose band follows centre(t) Hz: a time-varying band-pass done frame by frame in the frequency domain."""
    rng = np.random.default_rng(seed)
    win, hop = 1024, 256
    x = rng.standard_normal(n + win)
    w = np.hanning(win)
    out = np.zeros(n + win)
    norm = np.zeros(n + win)
    lf = np.log2(np.maximum(np.fft.rfftfreq(win, 1 / SR), 20))
    for i in range(0, n, hop):
        gain = np.exp(-.5 * ((lf - math.log2(max(20.0, centre(i / SR)))) / width) ** 2)
        out[i:i + win] += np.fft.irfft(np.fft.rfft(x[i:i + win] * w) * gain, win) * w
        norm[i:i + win] += w * w
    return (out / np.maximum(norm, 1e-3))[:n]


def _unit(x):
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return x / peak if peak > 0 else x


def _fade(x, ms_in, ms_out):
    a, b = int(SR * ms_in / 1000), int(SR * ms_out / 1000)
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b)
    return x


def _tone(freq):
    """Sine that follows an array of instantaneous frequencies."""
    return np.sin(2 * np.pi * np.cumsum(freq) / SR)


def _whoosh(d=.6):
    n = int(d * SR)
    u = np.arange(n) / n
    body = _shaped_noise(n, lambda s: 380 + 2600 * math.sin(math.pi * min(1.0, s / d) ** .7) ** 2, .6, 11)
    return _fade(_unit(body) * np.sin(np.pi * u ** .7) ** 2, 2, 20)


def _swell(d=.7):
    """Reverse-cymbal style rise; placed so that its END lands on the beat."""
    n = int(d * SR)
    u = np.arange(n) / n
    body = _shaped_noise(n, lambda s: 500 + 5200 * (s / d) ** 2, .9, 12)
    return _fade(_unit(body) * u ** 3, 1, 4)


def _hit(d=.45):
    n = int(d * SR)
    t = np.arange(n) / SR
    body = _tone(45 + 95 * np.exp(-t / .035)) * np.exp(-t / .13)
    mid = _tone(150 + 120 * np.exp(-t / .02)) * np.exp(-t / .05)
    click = _unit(_shaped_noise(n, lambda s: 2500, 1.2, 13)) * np.exp(-t / .008)
    snap = _unit(_shaped_noise(n, lambda s: 900, 1.0, 16)) * np.exp(-t / .03)     # what a phone speaker can actually play
    return _fade(np.tanh(2.2 * (body + .45 * mid + .5 * click + .35 * snap)), .5, 30)


def _boom(d=1.8):
    n = int(d * SR)
    t = np.arange(n) / SR
    f = 30 + 55 * np.exp(-t / .18)
    body = _tone(f) * np.exp(-t / .55) + .5 * _tone(2 * f) * np.exp(-t / .2)
    rumble = _unit(_shaped_noise(n, lambda s: 110, 1.0, 14)) * np.exp(-t / .4)
    click = _unit(_shaped_noise(n, lambda s: 1800, 1.3, 15)) * np.exp(-t / .012)
    knock = _unit(_shaped_noise(n, lambda s: 420, .8, 17)) * np.exp(-t / .15)     # mid-range body for small speakers
    return _fade(np.tanh(1.8 * (body + .55 * rumble + .35 * click + .4 * knock)), .5, 80)


def _pop(d=.14):
    n = int(d * SR)
    t = np.arange(n) / SR
    return _fade(_tone(380 + 900 * (1 - np.exp(-t / .012))) * np.exp(-t / .03), .3, 15)


SOUNDS = {'whoosh': _whoosh, 'swell': _swell, 'hit': _hit, 'boom': _boom, 'pop': _pop}


@functools.lru_cache(maxsize=None)
def sound(name):
    """One synthesised effect as float32 mono at SR, peak 0.89 (-1 dBFS)."""
    x = _unit(SOUNDS[name]()) * .89
    return x.astype(np.float32)


def write_wav(path, samples):
    pcm = (np.clip(samples, -1, 1) * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- where things happen
def loudness(src, start, length):
    """Loudness of the clip in dB, one value per HOP seconds (empty when there is no sound)."""
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{max(0.0, start):.3f}', '-t', f'{length:.3f}', '-i', str(src),
           '-vn', '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1']
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=180).stdout
    except (OSError, subprocess.SubprocessError):
        return np.zeros(0)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    size = int(16000 * HOP)
    frames = len(x) // size
    if frames < 4:
        return np.zeros(0)
    rms = np.sqrt(np.mean(x[:frames * size].reshape(frames, size) ** 2, axis=1))
    return 20 * np.log10(rms + 1e-6)


def _weight(words, i):
    import excite
    return max(excite.word_weight(words[i]['text']), excite._phrase_weight(words, i))


def emphasis(env, words, duration):
    """Per-frame 'something happens here' score: loudness above the clip's own level plus shouted / exclaimed words."""
    n = max(1, int(math.ceil(duration / HOP)))
    score = np.zeros(n)
    if len(env) >= 4:
        e = np.interp(np.arange(n), np.arange(len(env)), env)
        k = max(1, int(round(.25 / HOP)))
        e = np.convolve(e, np.ones(k) / k, 'same')
        spread = float(np.percentile(e, 90) - np.percentile(e, 50)) or 1.0
        score += np.clip((e - np.median(e)) / spread, -1, 3)
    t = np.arange(n) * HOP
    for i, w in enumerate(words or []):
        weight = _weight(words, i)
        if weight >= 1:
            score += weight / 3 * np.exp(-.5 * ((t - float(w['start'])) / .35) ** 2)
    return score


def peaks(score, lo, hi, gap, count, exclude=()):
    """Up to `count` of the best moments (seconds) inside [lo, hi], at least `gap` apart and outside the excluded spans."""
    out = []
    for i in np.argsort(-score, kind='stable'):
        t = round(float(i) * HOP, 3)
        if not lo <= t <= hi or any(a <= t <= b for a, b in exclude):
            continue
        if all(abs(t - p) >= gap for p in out):
            out.append(t)
            if len(out) >= count:
                break
    return sorted(out)


def choose(score, duration, opts):
    """Pick the slow-motion window and the punch-in zooms (source seconds, relative to the clip start)."""
    slow, zooms = None, []
    if opts.get('slowmo') and duration >= 8:
        p = peaks(score, 2.0, duration - 2.6, 1, 1)
        if p:
            a = round(max(1.0, p[0] - .45), 3)
            b = round(min(duration - 1.4, a + SLOW_LEN), 3)
            if b - a >= .8:
                slow = (a, b)
    if opts.get('zoom') and duration >= 5:
        count = max(1, min(4, int(round(duration / 16))))
        keep_out = [(slow[0] - 1.8, slow[1] + 1.2)] if slow else []
        for t in peaks(score, 1.6, duration - 1.8, 3.5, count, keep_out):
            t0 = round(max(.2, t - .08), 3)
            t1 = round(min(duration - .4, t0 + 1.6), 3)
            if slow and t0 < slow[0] < t1:
                t1 = round(slow[0] - .3, 3)
            if t1 - t0 >= .6:
                zooms.append((t0, t1))
    return slow, zooms


def _probe(src):
    """(frame rate as an FFmpeg rate string, its value, has_audio) of the source."""
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,r_frame_rate,avg_frame_rate', '-of', 'json', str(src)],
                           capture_output=True, text=True, timeout=30)
        streams = json.loads(r.stdout or '{}').get('streams', [])
    except (OSError, ValueError, subprocess.SubprocessError):
        streams = []
    rate, value, audio, found = '30', 30.0, False, False
    for s in streams:
        if s.get('codec_type') == 'audio':
            audio = True
        elif s.get('codec_type') == 'video' and not found:
            for key in ('avg_frame_rate', 'r_frame_rate'):
                try:
                    a, b = str(s.get(key, '')).split('/')
                    v = float(a) / float(b)
                except (ValueError, ZeroDivisionError):
                    continue
                if 10 <= v <= 120:
                    rate, value, found = f'{int(a)}/{int(b)}', v, True
                    break
    return rate, value, audio


# ---------------------------------------------------------------- the plan the export splices in
def _ease(x):
    """Cosine ease 0..1 of an FFmpeg expression already clipped to 0..1."""
    return f'(0.5-0.5*cos(PI*{x}))'


class Plan:
    def __init__(self, opts, source_duration, slow, zooms, words, width, height, rate, rate_value, has_audio, layered=False, post=True, center=None):
        self.opts, self.slow, self.width, self.height = opts, slow, width, height
        self.rate, self.rate_value, self.has_audio, self.post = rate, rate_value, has_audio, post
        self.amount = LAYERED_ZOOM if layered else 1.0
        self.center = center or (width / 2, height / 2)
        self.source_duration = source_duration
        self.duration = round(self.retime(source_duration), 3)
        self.zooms = [(round(self.retime(a), 3), round(self.retime(b), 3)) for a, b in zooms]
        self.words = [dict(w, start=round(self.retime(w['start']), 3), end=round(self.retime(w['end']), 3)) for w in words or []]
        self.cues = []
        self.inputs = []
        self.bed = None

    # ---- time
    def retime(self, t):
        """Source seconds -> output seconds (the slow-motion window plays 1/SLOW times longer)."""
        t = float(t)
        if not self.slow:
            return t
        a, b = self.slow
        if t <= a:
            return t
        if t <= b:
            return a + (t - a) / SLOW
        return t + (b - a) * (1 / SLOW - 1)

    @property
    def slow_out(self):
        return (self.retime(self.slow[0]), self.retime(self.slow[1])) if self.slow else None

    # ---- sound effects
    def plan_sfx(self, source_words, score):
        if not self.opts.get('sfx'):
            return
        cues = [('whoosh', 0.0)]
        for t0, _ in self.zooms:
            cues.append(('hit', max(0.0, t0 - .02)))
        if self.slow:
            a, b = self.slow_out
            cues += [('swell', max(0.0, a - len(sound('swell')) / SR)), ('boom', a), ('whoosh', max(0.0, b - .3))]
        busy = [t for _, t in cues] + ([self.slow_out[0] + .7] if self.slow else [])
        strong = []
        for i, w in enumerate(source_words or []):
            weight = _weight(source_words, i)
            if weight >= 2:
                at = float(w['start'])
                frame = min(len(score) - 1, max(0, int(at / HOP))) if len(score) else 0
                strong.append((weight + (float(score[frame]) if len(score) else 0), self.retime(at)))
        pops = []
        for _, at in sorted(strong, key=lambda x: -x[0]):
            if self.slow and self.slow_out[0] - .5 <= at <= self.slow_out[1] + .3:
                continue
            if .8 <= at <= self.duration - .5 and all(abs(at - t) >= 1.0 for t in busy) and all(abs(at - t) >= 2.5 for t in pops):
                pops.append(at)
                if len(pops) >= 3:
                    break
        cues += [('pop', t) for t in pops]
        self.cues = sorted(((name, round(t, 3)) for name, t in cues if t < self.duration - .1), key=lambda c: (c[1], c[0]))

    def write_bed(self, path):
        """Render every cue into one mono WAV the length of the clip; it becomes one extra FFmpeg input."""
        if not self.cues or self.opts.get('sfx_volume', 1.0) <= 0:
            return None
        n = int(math.ceil(self.duration * SR)) + SR // 10
        bed = np.zeros(n, dtype=np.float32)
        for name, t in self.cues:
            snd = sound(name) * (GAIN[name] * self.opts.get('sfx_volume', 1.0))
            i = int(max(0.0, t) * SR)
            if i < n:
                part = snd[:n - i]
                bed[i:i + len(part)] += part
        write_wav(path, bed)
        self.bed = Path(path)
        self.inputs = ['-i', str(path)]
        return self.bed

    # ---- video
    def _retime_graph(self):
        # The slow part keeps every source frame (a 60 fps source at half speed is a real 30 fps); `framerate` only blends
        # in-between frames when the slowed footage has fewer frames than the output rate.
        a, b = self.slow
        r = self.rate
        g = (f'[0:v]split=3[fxv0][fxv1][fxv2];'
             f'[fxv0]trim=end={a:.3f},setpts=PTS-STARTPTS,fps={r}[fxp0];'
             f'[fxv1]trim=start={a:.3f}:end={b:.3f},setpts=(PTS-STARTPTS)/{SLOW},framerate=fps={r}[fxp1];'
             f'[fxv2]trim=start={b:.3f},setpts=PTS-STARTPTS,fps={r}[fxp2]')
        if not self.has_audio:
            return g + ';[fxp0][fxp1][fxp2]concat=n=3:v=1:a=0[fxsrc]'
        g += (f';[0:a]aresample={SR},asplit=3[fxa0][fxa1][fxa2];'
              f'[fxa0]atrim=end={a:.3f},asetpts=PTS-STARTPTS[fxq0];'
              f'[fxa1]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,asetrate={int(SR * PITCH)},aresample={SR},atempo={SLOW / PITCH:.4f}[fxq1];'
              f'[fxa2]atrim=start={b:.3f},asetpts=PTS-STARTPTS[fxq2]')
        return g + ';[fxp0][fxq0][fxp1][fxq1][fxp2][fxq2]concat=n=3:v=1:a=1[fxsrc][fxa]'

    def _post_chain(self):
        """Zoom + shake (one perspective filter, only active inside its windows), vignette and flash for the slow beat."""
        T = f'((in-1)/{self.rate_value:.6f})'
        terms, spans, dx, dy = [], [], '0', '0'
        for t0, t1 in self.zooms:
            z = .2 * self.amount
            rise = f'clip(({T}-{t0:.3f})/0.12,0,1)'
            fall = f'clip(({T}-{t1 - .22:.3f})/0.22,0,1)'
            terms.append(f'{z:.3f}*{_ease(rise)}*(1-{_ease(fall)})')
            spans.append((t0 - .05, t1 + .05))
        chain = []
        if self.slow:
            A, B = self.slow_out
            push = f'clip(({T}-{A:.3f})/{B - A:.3f},0,1)'
            snap = f'clip(({T}-{B:.3f})/0.15,0,1)'
            terms.append(f'{.12 * self.amount:.3f}*{_ease(push)}*(1-{_ease(snap)})')
            decay = f'exp(-max({T}-{A:.3f},0)/0.12)*between({T},{A:.3f},{A + .45:.3f})'
            terms.append(f'0.05*{decay}')
            shake = 14 * self.width / 1080
            dx = f'{shake:.1f}*sin(107*({T}-{A:.3f}))*{decay}'
            dy = f'{shake * .8:.1f}*cos(83*({T}-{A:.3f}))*{decay}'
            spans.append((A - .05, B + .25))
        if terms:
            W, H = self.width, self.height
            cx, cy = self.center
            z = '(1+' + '+'.join(terms) + ')'
            x0, y0 = f'{cx:.1f}-{cx:.1f}/{z}+{dx}', f'{cy:.1f}-{cy:.1f}/{z}+{dy}'
            x1, y1 = f'{cx:.1f}+{W - cx:.1f}/{z}+{dx}', f'{cy:.1f}+{H - cy:.1f}/{z}+{dy}'
            enable = '+'.join(f'between(t,{max(0.0, a):.3f},{b:.3f})' for a, b in spans)
            chain.append(f"perspective=x0='{x0}':y0='{y0}':x1='{x1}':y1='{y0}':x2='{x0}':y2='{y1}':x3='{x1}':y3='{y1}'"
                         f":interpolation=linear:eval=frame:enable='{enable}'")
        if self.slow:
            A, B = self.slow_out
            chain.append(f"vignette=angle=PI/4:enable='between(t,{A:.3f},{B:.3f})'")
            chain.append(f"eq=brightness='0.45*exp(-(t-{A:.3f})/0.08)':eval=frame:enable='between(t,{A:.3f},{A + .35:.3f})'")
        return chain

    def video(self, graph):
        """-> (graph, label of the finished picture before captions). Slow motion replaces the source input [0:v]."""
        if self.slow:
            graph = self._retime_graph() + ';' + graph.replace('[0:v]', '[fxsrc]')
        chain = self._post_chain() if self.post else []
        if not chain:
            return graph, '[base]'
        return graph + f';[base]fps={self.rate},' + ','.join(chain) + '[fxbase]', '[fxbase]'

    # ---- audio
    def voice(self, music_graph):
        """The music bed (styling.Plan.audio) mixes the voice from [0:a]; inside slow motion it must take the retimed voice."""
        return music_graph.replace('[0:a]', '[fxa]') if self.slow and self.has_audio else music_graph

    def audio(self, amap, index, fade=False):
        """-> (graph piece, label to map or None). `amap` is the music mix label if there is one, `index` the bed's input number."""
        voice = amap or ('[fxa]' if self.slow and self.has_audio else ('[0:a]' if self.has_audio else None))
        parts, label = [], voice
        if self.bed:
            parts.append(f'[{index}:a]aformat=sample_rates={SR}:channel_layouts=stereo[fxbed]')
            if voice:
                parts.append(f'{voice}aformat=sample_rates={SR}:channel_layouts=stereo[fxvoice]')
                parts.append('[fxvoice][fxbed]amix=inputs=2:duration=first:normalize=0:dropout_transition=0,alimiter=limit=0.95:level=0[fxmix]')
            else:
                parts.append('[fxbed]anull[fxmix]')
            label = '[fxmix]'
        if fade and label and (parts or label != amap):
            parts.append(f'{label}afade=t=in:st=0:d=0.25,afade=t=out:st={max(0.0, self.duration - .3):.2f}:d=0.3[fxfade]')
            label = '[fxfade]'
        if not parts and label == amap:
            return '', amap
        return ';'.join(parts), label

    def summary(self):
        return {'slowmo': list(self.slow_out) if self.slow else None, 'zooms': [list(z) for z in self.zooms],
                'sfx': [[name, t] for name, t in self.cues], 'duration': self.duration}


FACE_LAYOUTS = ('stack', 'bottom', 'circle')     # punch in on the speaker's face; other layouts zoom on the picture centre


def zoom_center(layout, top, width, height):
    """Where punch-ins zoom to: the facecam layer's centre for face-on-top/bottom layouts, otherwise the frame centre."""
    if layout in FACE_LAYOUTS:
        try:
            from layouts import geometry
            cam = next((l for l in geometry(layout, top, width, height)['layers'] if l['source'] == 'cam'), None)
            if cam:
                return cam['x'] + cam['w'] / 2, cam['y'] + cam['h'] / 2
        except (ValueError, TypeError):
            pass
    return width / 2, height / 2


def build(body, src, start, end, words, width, height, layout, data, job, template='none', top=.32, fps=0):
    """Analyse the clip and return a Plan, or None when no effect is switched on. `fps` is the export frame rate (0 = source)."""
    opts = settings(body)
    if not (opts['sfx'] or opts['zoom'] or opts['slowmo']):
        return None
    length = float(end) - float(start)
    env = loudness(src, start, length)
    score = emphasis(env, words, length)
    slow, zooms = choose(score, length, opts)
    rate, value, audio = _probe(src)
    if fps:
        rate, value = str(int(fps)), float(fps)
    layered = layout not in ('blur', 'fill', 'game')
    plan = Plan(opts, length, slow, zooms, words, width, height, rate, value, audio, layered, post=template == 'none',
                center=zoom_center(layout, top, width, height))
    plan.plan_sfx(words, score)
    plan.write_bed(Path(data) / f'{job}.sfx.wav')
    return plan
