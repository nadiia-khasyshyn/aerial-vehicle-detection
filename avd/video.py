"""Video writing, shared by the torch and the onnxruntime entry points."""

import subprocess


class FragmentedWriter:
    """Fragmented MP4 via ffmpeg: an interrupted run still leaves a playable file."""

    def __init__(self, path, fps, size):
        w, h = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "veryfast", "-crf", "23",
             "-g", "30", "-frag_duration", "1000000", "-flush_packets", "1",
             "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
             str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def release(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.proc.wait()


def even_size(width, height, max_width):
    """Output size for yuv420p, which rejects an odd dimension outright."""
    scale = max_width / width if width > max_width else 1.0
    return int(width * scale) // 2 * 2, int(height * scale) // 2 * 2
