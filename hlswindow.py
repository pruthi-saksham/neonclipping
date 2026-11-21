"""Cut a time window out of a long HLS VOD (Twitch) by copying only the segments that cover it.

yt-dlp hands a section download to FFmpeg, which seeks inside the HLS playlist by timestamp. MPEG-TS timestamps wrap every
2^33 / 90 kHz = 26.5 hours, so on longer VODs (Twitch marathons run 24-48 h) a seek past that point keeps nothing useful and
the section comes out empty or cut short. Here the playlist is read directly, the segments covering the window are written
into a small local playlist, and FFmpeg copies just those: no long seek happens, whatever the length of the VOD.
"""
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


def parse(text, base):
    """[(start_seconds, duration, absolute_url)] of an HLS media playlist."""
    out, t, dur = [], 0.0, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('#EXTINF:'):
            dur = float(line[8:].split(',')[0])
        elif line and not line.startswith('#') and dur is not None:
            out.append((t, dur, urllib.parse.urljoin(base, line)))
            t += dur
            dur = None
    return out


def window(segments, start, end):
    """The segments that overlap [start, end)."""
    return [s for s in segments if s[0] + s[1] > start and s[0] < end]


def pick(info, max_height, lowest=False):
    """The HLS format that holds picture AND sound (Twitch's renditions do), best within max_height (or the smallest)."""
    usable = [f for f in info.get('formats') or [] if str(f.get('protocol', '')).startswith('m3u8') and f.get('url')
              and f.get('vcodec') not in (None, 'none') and f.get('acodec') not in (None, 'none') and (f.get('height') or 0) <= max_height]
    if not usable:
        return None
    usable.sort(key=lambda f: ((f.get('height') or 0), (f.get('fps') or 0), (f.get('tbr') or 0)))
    return usable[0] if lowest else usable[-1]


def _get(url, headers, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=dict(headers or {})), timeout=timeout).read()


def fetch(playlist_url, start, end, out, headers=None):
    """Copy the segments covering [start, end] of the playlist into `out` (MP4). Returns the source second where `out` begins
    (the start of its first segment, at most one segment before `start`). The segments are downloaded here and FFmpeg only
    joins local files, so no seek (and no timestamp wrap) is involved."""
    segs = window(parse(_get(playlist_url, headers).decode('utf-8', 'replace'), playlist_url), start, end)
    if not segs:
        raise RuntimeError('That part of the video is not in its playlist.')
    out = Path(out)
    parts = []
    try:
        for n, (_, _, url) in enumerate(segs):
            part = out.with_name(f'{out.stem}.part{n:03d}.ts')
            part.write_bytes(_get(url, headers, timeout=120))
            parts.append(part)
        r = subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', 'concat:' + '|'.join(str(p) for p in parts),
                            '-map', '0:v:0?', '-map', '0:a:0?', '-c', 'copy', '-movflags', '+faststart', str(out)],
                           capture_output=True, text=True, timeout=1800)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)
    if r.returncode:
        raise RuntimeError('Could not join the video segments: ' + r.stderr[-300:])
    return segs[0][0]


def usable(path, min_seconds=1.0):
    """True when the file has a picture stream and a real duration (an empty or cut-off section is not usable)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size < 1024:
        return False
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_type:format=duration',
                            '-of', 'default=nw=1', str(p)], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    if 'codec_type=video' not in r.stdout:
        return False
    try:
        return float(r.stdout.split('duration=')[1].split()[0]) >= min_seconds
    except (IndexError, ValueError):
        return False
