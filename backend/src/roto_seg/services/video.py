"""
Video Processing Pipeline.

Handles video I/O, frame extraction, and processing coordination.
Optimized for memory efficiency on M1 Macs.
"""

import gc
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, List, Union
import numpy as np

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

import cv2
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Video metadata."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # seconds
    codec: str


class VideoReader:
    """
    Memory-efficient video reader.

    Uses PyAV for efficient video decoding.
    Falls back to OpenCV if PyAV is not available.
    """

    def __init__(self, path: str):
        """
        Initialize video reader.

        Args:
            path: Path to video file
        """
        self.path = str(path)
        self._container = None
        self._stream = None
        self._info: Optional[VideoInfo] = None

    @property
    def info(self) -> VideoInfo:
        """Get video information."""
        if self._info is None:
            self._info = self._get_video_info()
        return self._info

    def _get_video_info(self) -> VideoInfo:
        """Extract video metadata."""
        if AV_AVAILABLE:
            container = av.open(self.path)
            stream = container.streams.video[0]

            info = VideoInfo(
                path=self.path,
                width=stream.width,
                height=stream.height,
                fps=float(stream.average_rate),
                frame_count=stream.frames or int(stream.duration * stream.time_base * float(stream.average_rate)),
                duration=float(stream.duration * stream.time_base) if stream.duration else 0,
                codec=stream.codec_context.name,
            )
            container.close()
            return info
        else:
            # Fallback to OpenCV
            cap = cv2.VideoCapture(self.path)
            info = VideoInfo(
                path=self.path,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=cap.get(cv2.CAP_PROP_FPS),
                frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                duration=cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                codec="unknown",
            )
            cap.release()
            return info

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Read a single frame.

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            RGB frame as numpy array (H, W, 3) or None if failed
        """
        if AV_AVAILABLE:
            return self._read_frame_av(frame_idx)
        else:
            return self._read_frame_cv2(frame_idx)

    def _read_frame_av(self, frame_idx: int) -> Optional[np.ndarray]:
        """Read frame using PyAV."""
        container = av.open(self.path)
        stream = container.streams.video[0]

        # Seek to frame
        pts = int(frame_idx * stream.duration / stream.frames) if stream.frames else frame_idx
        container.seek(pts, stream=stream)

        # Read frame
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format='rgb24')
            container.close()
            return img

        container.close()
        return None

    def _read_frame_cv2(self, frame_idx: int) -> Optional[np.ndarray]:
        """Read frame using OpenCV."""
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        cap.release()

        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def iter_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterate through frames efficiently.

        Args:
            start: Starting frame index
            end: Ending frame index (exclusive). None = all frames
            step: Frame step (1 = every frame, 2 = every other frame)

        Yields:
            Tuple of (frame_index, frame_array)
        """
        if end is None:
            end = self.info.frame_count

        if AV_AVAILABLE:
            yield from self._iter_frames_av(start, end, step)
        else:
            yield from self._iter_frames_cv2(start, end, step)

    def _iter_frames_av(
        self,
        start: int,
        end: int,
        step: int,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterate frames using PyAV."""
        container = av.open(self.path)
        stream = container.streams.video[0]

        # Seek to start
        if start > 0 and stream.frames:
            pts = int(start * stream.duration / stream.frames)
            container.seek(pts, stream=stream)

        frame_idx = start
        for frame in container.decode(video=0):
            if frame_idx >= end:
                break

            if (frame_idx - start) % step == 0:
                img = frame.to_ndarray(format='rgb24')
                yield frame_idx, img

            frame_idx += 1

            # Periodic garbage collection for memory management
            if frame_idx % 100 == 0:
                gc.collect()

        container.close()

    def _iter_frames_cv2(
        self,
        start: int,
        end: int,
        step: int,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterate frames using OpenCV."""
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frame_idx = start
        while frame_idx < end:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start) % step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_idx, rgb_frame

            frame_idx += 1

            if frame_idx % 100 == 0:
                gc.collect()

        cap.release()


class ImageSequenceReader:
    """
    Read image sequences (EXR, PNG, DPX, etc.).
    """

    SUPPORTED_EXTENSIONS = {'.exr', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.dpx'}

    def __init__(self, path: str, pattern: Optional[str] = None):
        """
        Initialize sequence reader.

        Args:
            path: Path to first frame or directory
            pattern: Optional filename pattern (e.g., "frame.%04d.exr")
        """
        self.path = Path(path)
        self.pattern = pattern
        self._frames: List[Path] = []
        self._info: Optional[VideoInfo] = None

        self._discover_frames()

    def _discover_frames(self):
        """Find all frames in sequence."""
        if self.path.is_file():
            # Single file - find sequence from naming pattern
            directory = self.path.parent
            stem = self.path.stem
            ext = self.path.suffix

            # Find frame number in filename
            import re
            match = re.search(r'(\d+)$', stem)
            if match:
                padding = len(match.group(1))
                prefix = stem[:match.start()]

                # Find all matching files
                for f in sorted(directory.iterdir()):
                    if f.suffix == ext and f.stem.startswith(prefix):
                        self._frames.append(f)
            else:
                self._frames = [self.path]

        elif self.path.is_dir():
            # Directory - find all image files
            for ext in self.SUPPORTED_EXTENSIONS:
                self._frames.extend(sorted(self.path.glob(f'*{ext}')))

    @property
    def info(self) -> VideoInfo:
        """Get sequence information."""
        if self._info is None:
            if not self._frames:
                raise ValueError("No frames found in sequence")

            # Read first frame for dimensions
            first_frame = cv2.imread(str(self._frames[0]))
            h, w = first_frame.shape[:2]

            self._info = VideoInfo(
                path=str(self.path),
                width=w,
                height=h,
                fps=24.0,  # Default, can be overridden
                frame_count=len(self._frames),
                duration=len(self._frames) / 24.0,
                codec="image_sequence",
            )

        return self._info

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Read a single frame."""
        if frame_idx < 0 or frame_idx >= len(self._frames):
            return None

        frame = cv2.imread(str(self._frames[frame_idx]), cv2.IMREAD_UNCHANGED)
        if frame is None:
            return None

        # Handle different formats
        if len(frame.shape) == 2:
            # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            # BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def iter_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterate through frames."""
        if end is None:
            end = len(self._frames)

        for i in range(start, min(end, len(self._frames)), step):
            frame = self.read_frame(i)
            if frame is not None:
                yield i, frame

            if i % 100 == 0:
                gc.collect()


def get_reader(path: str) -> Union[VideoReader, ImageSequenceReader]:
    """
    Get appropriate reader for the given path.

    Args:
        path: Path to video file or image sequence

    Returns:
        VideoReader or ImageSequenceReader
    """
    path_obj = Path(path)

    # Check if it's a video file
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mxf', '.webm'}

    if path_obj.suffix.lower() in video_extensions:
        return VideoReader(path)
    elif path_obj.suffix.lower() in ImageSequenceReader.SUPPORTED_EXTENSIONS:
        return ImageSequenceReader(path)
    elif path_obj.is_dir():
        return ImageSequenceReader(path)
    else:
        # Default to video reader
        return VideoReader(path)
