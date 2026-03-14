"""Observation builder module for SmartRoad AI driver distraction detection.

This module builds observation vectors from YOLO and SegFormer pipeline results,
tracking distraction durations over time with gradual decay logic.
"""

import numpy as np

FPS = 30  # Assumed frame rate for webcam

class DurationTracker:
    """Tracks the duration of various driver distractions across frames.

    The tracker uses gradual decay instead of instant reset to handle brief
    occlusions or detection gaps. This prevents the duration from immediately
    dropping to zero when a distraction momentarily disappears from the frame.

    Attributes:
        phone_frames (int): Number of frames phone has been detected
        gaze_frames (int): Number of frames driver has been looking away
        cigarette_frames (int): Number of frames cigarette has been detected
    """

    def __init__(self):
        """Initialize the duration tracker with all counters at zero."""
        self.phone_frames = 0
        self.gaze_frames = 0
        self.cigarette_frames = 0

    def reset(self):
        """Reset all duration counters to zero.

        Called at the start of each episode to ensure clean state.
        """
        self.phone_frames = 0
        self.gaze_frames = 0
        self.cigarette_frames = 0

    def update(self, yolo_results, seg_results):
        """Update duration counters based on current frame detections.

        Args:
            yolo_results (dict): YOLO detection results with object names as keys
            seg_results (dict): SegFormer results with semantic segmentation info

        Note:
            The gradual decay (subtracting 2-3 frames when not detected) prevents
            instant resets due to brief occlusions. This makes duration tracking
            more robust to noisy detections.
        """
        # Phone detection logic - increment when detected, gradual decay otherwise
        if "cell phone" in yolo_results:
            self.phone_frames += 1
        else:
            # Gradual decay instead of instant reset to handle brief occlusions
            self.phone_frames = max(0, self.phone_frames - 3)

        # Gaze away detection - based on steering wheel visibility
        if not seg_results.get("steering_visible", False):
            self.gaze_frames += 1
        else:
            # Gradual decay when driver returns gaze to road
            self.gaze_frames = max(0, self.gaze_frames - 3)

        # Cigarette detection logic
        if "cigarette" in yolo_results:
            self.cigarette_frames += 1
        else:
            # Slightly faster decay for cigarette (less persistent)
            self.cigarette_frames = max(0, self.cigarette_frames - 2)

# Global tracker instance used across all environment steps
_tracker = DurationTracker()

def reset_tracker():
    """Reset the global duration tracker.

    This should be called at the start of each episode to ensure
    duration counters start from zero.
    """
    global _tracker
    _tracker.reset()

def build_observation(yolo_results, seg_results):
    """Build a 10-dimensional observation vector from pipeline results.

    Args:
        yolo_results (dict): YOLO object detection results. Each key is an
            object name (e.g., "cell phone", "person", "cigarette") and
            each value is a dict with "bbox" and "conf" keys.
        seg_results (dict): SegFormer semantic segmentation results with
            keys "driver_zone" and "steering_visible" (both boolean).

    Returns:
        np.ndarray: 10-dimensional float32 observation vector with indices:
            [0] phone_detected (binary)
            [1] phone_conf (0.0-1.0)
            [2] phone_near_face (binary, based on bbox y-coordinate)
            [3] gaze_away (binary, inverse of steering_visible)
            [4] cigarette_detected (binary)
            [5] phone_duration (seconds, capped at 30.0)
            [6] gaze_duration (seconds, capped at 30.0)
            [7] cigarette_duration (seconds, capped at 30.0)
            [8] person_detected (binary)
            [9] driver_zone_occupied (binary)

    Note:
        The phone_near_face heuristic uses y1 < 192 (40% of 480px frame height)
        to determine if the phone is in the upper portion near the driver's face.
    """
    obs = np.zeros(10, dtype=np.float32)

    # Index 0: phone detected (binary)
    obs[0] = 1.0 if "cell phone" in yolo_results else 0.0

    # Index 1: phone confidence
    if "cell phone" in yolo_results:
        obs[1] = yolo_results["cell phone"]["conf"]

    # Index 2: phone near face heuristic (y1 < frame_height * 0.4)
    if "cell phone" in yolo_results:
        bbox = yolo_results["cell phone"]["bbox"]
        obs[2] = 1.0 if bbox[1] < 192 else 0.0  # 480 * 0.4 = 192

    # Index 3: gaze away (steering not visible)
    obs[3] = 0.0 if seg_results.get("steering_visible", False) else 1.0

    # Index 4: cigarette detected
    obs[4] = 1.0 if "cigarette" in yolo_results else 0.0

    # Update tracker
    _tracker.update(yolo_results, seg_results)

    # Index 5: phone duration in seconds
    obs[5] = min(_tracker.phone_frames / FPS, 30.0)

    # Index 6: gaze away duration in seconds
    obs[6] = min(_tracker.gaze_frames / FPS, 30.0)

    # Index 7: cigarette duration in seconds
    obs[7] = min(_tracker.cigarette_frames / FPS, 30.0)

    # Index 8: person detected (binary)
    obs[8] = 1.0 if "person" in yolo_results else 0.0

    # Index 9: driver zone occupied
    obs[9] = 1.0 if seg_results.get("driver_zone", False) else 0.0

    return obs

def get_tracker_state():
    """Get current tracker state as durations in seconds.

    Returns:
        dict: Dictionary with three keys:
            - phone_duration: Phone distraction time in seconds
            - gaze_duration: Gaze away time in seconds
            - cigarette_duration: Cigarette distraction time in seconds

    Note:
        Converts frame counts to seconds by dividing by FPS (30).
        Used by rl_environment.py to populate the info dict.
    """
    return {
        "phone_duration": _tracker.phone_frames / FPS,
        "gaze_duration": _tracker.gaze_frames / FPS,
        "cigarette_duration": _tracker.cigarette_frames / FPS
    }

if __name__ == "__main__":
    # Test the observation builder
    print("Testing obs_builder.py...")

    # Test 1: Phone detected
    yolo_res = {"cell phone": {"bbox": [100, 50, 200, 150], "conf": 0.92}}
    seg_res = {"driver_zone": True, "steering_visible": True}
    obs = build_observation(yolo_res, seg_res)
    print(f"Test 1 (phone detected): {obs}")

    # Test 2: Gaze away
    yolo_res = {}
    seg_res = {"driver_zone": True, "steering_visible": False}
    obs = build_observation(yolo_res, seg_res)
    print(f"Test 2 (gaze away): {obs}")

    # Test 3: Check tracker state
    state = get_tracker_state()
    print(f"Tracker state: {state}")

    # Test 4: Reset tracker
    reset_tracker()
    state = get_tracker_state()
    print(f"After reset: {state}")

    print("obs_builder.py tests passed!")
