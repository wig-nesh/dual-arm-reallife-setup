import os
import threading
import time

import cv2

from realsense.src.realsense_camera import RealSenseCamera


class RealSenseVideoRecorder:
    def __init__(self, output_path, width=640, height=480, fps=30):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = RealSenseCamera(width=width, height=height, fps=fps)
        self.writer = None
        self._stop = threading.Event()
        self._thread = None
        self._frame_count = 0

    def start(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.camera.start()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )
        if not self.writer.isOpened():
            self.camera.stop()
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"Recording RealSense video to: {self.output_path}")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.writer is not None:
            self.writer.release()
        self.camera.stop()
        print(
            f"Stopped RealSense recording: {self.output_path} "
            f"({self._frame_count} frames)"
        )

    def _run(self):
        frame_period = 1.0 / self.fps
        while not self._stop.is_set():
            start = time.time()
            color_frame, _ = self.camera.get_frames()
            if color_frame is not None:
                self.writer.write(color_frame)
                self._frame_count += 1

            elapsed = time.time() - start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
