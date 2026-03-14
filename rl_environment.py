"""Reinforcement learning environment for SmartRoad AI driver distraction detection.

This module defines the DriverEnv Gymnasium environment that uses YOLO + SegFormer
pipeline to observe driver behavior and learn optimal action policies through PPO.
"""

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from obs_builder import build_observation, reset_tracker, get_tracker_state


class FakePipeline:
    """Generates realistic synthetic pipeline data for testing without webcam.

    Used when pipeline_fn=None in DriverEnv to allow testing and training
    without requiring a physical webcam or live video feed.

    Attributes:
        rng: Random number generator (numpy Generator or RandomState)
    """

    def __init__(self, rng=None):
        """Initialize FakePipeline with a random number generator.

        Args:
            rng: Random number generator to use. If None, creates new RandomState.
        """
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self):
        """Generate one frame of synthetic pipeline data.

        Returns:
            tuple: (yolo_results, seg_results, frame) where:
                - yolo_results: dict of detected objects with bbox and conf
                - seg_results: dict with driver_zone and steering_visible
                - frame: numpy array (480x640x3) black image

        Note:
            Probabilities are chosen to match real-world driving scenarios:
            - 40% phone detection (moderate distraction rate)
            - 25% gaze away (driver occasionally checks mirrors/passengers)
            - 10% cigarette (less common distraction)
        """
        yolo_results = {}

        # Handle both RandomState (legacy) and Generator (modern numpy) objects
        if hasattr(self.rng, 'integers'):
            # Modern numpy Generator
            randint = lambda low, high: self.rng.integers(low, high)
            uniform = lambda low, high: self.rng.uniform(low, high)
            random = lambda: self.rng.random()
        else:
            # Legacy RandomState
            randint = lambda low, high: self.rng.randint(low, high)
            uniform = lambda low, high: self.rng.uniform(low, high)
            random = lambda: self.rng.random()

        # 40% chance of phone detection
        if random() < 0.4:
            # Random bbox in upper portion (near face)
            x1 = randint(50, 300)
            y1 = randint(20, 150)
            x2 = x1 + randint(80, 120)
            y2 = y1 + randint(100, 150)
            yolo_results["cell phone"] = {
                "bbox": [x1, y1, x2, y2],
                "conf": uniform(0.7, 0.98)
            }

        # 25% chance of gaze away
        steering_visible = random() > 0.25

        # 10% chance of cigarette
        if random() < 0.1:
            x1 = randint(200, 400)
            y1 = randint(150, 300)
            x2 = x1 + randint(30, 60)
            y2 = y1 + randint(40, 80)
            yolo_results["cigarette"] = {
                "bbox": [x1, y1, x2, y2],
                "conf": uniform(0.6, 0.9)
            }

        # Always include person in driver zone
        yolo_results["person"] = {
            "bbox": [100, 100, 400, 400],
            "conf": 0.95
        }

        seg_results = {
            "driver_zone": True,
            "steering_visible": steering_visible
        }

        # Black fake frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        return yolo_results, seg_results, frame


class DriverEnv(gym.Env):
    """Custom Gymnasium environment for driver distraction detection.

    The agent observes driver behavior through YOLO + SegFormer pipeline
    and chooses actions: ALL_CLEAR (0), MONITOR (1), or VIOLATION (2).

    Key design decisions:
    - Gradual duration decay prevents instant resets from brief occlusions
    - Reward function penalizes both false positives and missed violations
    - Small positive reward for correct ALL_CLEAR prevents excessive violations
    - Episodes terminate early on extreme distraction (25+ seconds)

    Attributes:
        action_space: Discrete(3) for ALL_CLEAR, MONITOR, VIOLATION
        observation_space: Box(10,) with bounds [0.0, 30.0]
        pipeline_fn: Function returning (yolo_results, seg_results, frame)
        max_steps: Maximum steps per episode before termination
        max_violations_per_episode: Max allowed false violations before termination
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, pipeline_fn=None, max_steps=200, max_violations_per_episode=10):
        """Initialize the DriverEnv environment.

        Args:
            pipeline_fn: Optional callable that returns (yolo_results, seg_results, frame).
                If None, uses FakePipeline for testing.
            max_steps: Maximum steps per episode (default 200)
            max_violations_per_episode: Max false violations allowed (default 10)
        """
        super().__init__()

        # Action space: 0=ALL_CLEAR, 1=MONITOR, 2=VIOLATION
        self.action_space = spaces.Discrete(3)

        # Observation space: 10-dim float32 vector
        # [phone_detected, phone_conf, phone_near_face, gaze_away, cigarette_detected,
        #  phone_duration, gaze_duration, cigarette_duration, person_detected, driver_zone]
        self.observation_space = spaces.Box(
            low=0.0, high=30.0, shape=(10,), dtype=np.float32
        )

        self.pipeline_fn = pipeline_fn
        self.max_steps = max_steps
        self.max_violations_per_episode = max_violations_per_episode

        # Episode tracking
        self._step_count = 0
        self._violation_count = 0
        self._current_frame = None
        self._last_obs = None
        self._fake_pipeline = None

        # Episode stats tracking
        self._total_violations_flagged = 0
        self._max_phone_duration_seen = 0.0
        self._max_gaze_duration_seen = 0.0
        self._total_reward = 0.0

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            tuple: (observation, info) where observation is the initial 10-dim vector
        """
        super().reset(seed=seed)

        # Reset all trackers and episode counters
        reset_tracker()
        self._step_count = 0
        self._violation_count = 0
        self._total_violations_flagged = 0
        self._max_phone_duration_seen = 0.0
        self._max_gaze_duration_seen = 0.0
        self._total_reward = 0.0

        # Initialize FakePipeline with seeded RNG if needed
        if self.pipeline_fn is None:
            self._fake_pipeline = FakePipeline(rng=self.np_random)

        # Get initial observation
        pipeline_fn = self.pipeline_fn if self.pipeline_fn else self._fake_pipeline
        yolo_results, seg_results, frame = pipeline_fn()

        self._current_frame = frame
        obs = build_observation(yolo_results, seg_results)
        self._last_obs = obs

        return obs, {}

    def step(self, action):
        """Execute one environment step with the given action.

        Args:
            action (int): Action to take (0=ALL_CLEAR, 1=MONITOR, 2=VIOLATION)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: 10-dim float32 array
                - reward: Float reward value
                - terminated: Boolean indicating episode end
                - truncated: Always False (not used)
                - info: Dict with step, violations, durations, attention_score_approx
        """
        self._step_count += 1

        # Get pipeline results from live webcam or FakePipeline
        pipeline_fn = self.pipeline_fn if self.pipeline_fn else self._fake_pipeline
        yolo_results, seg_results, frame = pipeline_fn()

        self._current_frame = frame
        obs = build_observation(yolo_results, seg_results)
        self._last_obs = obs

        # Get tracker state for reward calculation
        tracker_state = get_tracker_state()
        phone_duration = tracker_state["phone_duration"]
        gaze_duration = tracker_state["gaze_duration"]

        # Update episode stats
        self._max_phone_duration_seen = max(self._max_phone_duration_seen, phone_duration)
        self._max_gaze_duration_seen = max(self._max_gaze_duration_seen, gaze_duration)

        # Reward function logic - incentivizes correct action for current distraction level
        reward = 0.0

        # Define what constitutes an actual violation worth flagging
        # Phone > 3s or gaze away > 4s are serious safety concerns
        actual_violation = (phone_duration > 3.0) or (gaze_duration > 4.0)

        if action == 2:  # VIOLATION - agent flags serious distraction
            self._total_violations_flagged += 1
            if actual_violation:
                # Correct violation flag - highest reward for catching real danger
                reward = 10.0
            else:
                # False positive - penalize crying wolf, wastes operator attention
                reward = -5.0
                self._violation_count += 1  # Count as mistake

        elif action == 1:  # MONITOR - agent indicates concern but not critical yet
            if actual_violation:
                # Too cautious - should have flagged immediately
                reward = -2.0
            elif phone_duration > 1.0 or gaze_duration > 1.5:
                # Correct monitoring - minor distraction needs watching
                reward = 2.0
            else:
                # Unnecessary monitoring when driver is fine
                reward = -1.0

        else:  # ALL_CLEAR (action == 0) - agent says everything is fine
            if actual_violation:
                # Missed violation - very dangerous, large penalty
                reward = -8.0
                self._violation_count += 1  # Count as serious mistake
            elif phone_duration > 1.0 or gaze_duration > 1.5:
                # Should be monitoring but said all clear - minor penalty
                reward = -1.0
            else:
                # Correct all clear - small positive reward prevents over-flagging
                # This encourages the agent to use ALL_CLEAR when appropriate
                reward = 0.5

        self._total_reward += reward

        # Termination conditions
        terminated = False

        # Condition 1: Max steps reached
        if self._step_count >= self.max_steps:
            terminated = True

        # Condition 2: Too many violations
        if self._violation_count >= self.max_violations_per_episode:
            terminated = True

        # Condition 3: Extreme distraction (NEW)
        if phone_duration > 25.0 or gaze_duration > 25.0:
            terminated = True

        # Build info dict
        attention_score_approx = phone_duration * 0.4 + gaze_duration * 0.3

        info = {
            "step": self._step_count,
            "violations": self._violation_count,
            "phone_duration": phone_duration,
            "gaze_duration": gaze_duration,
            "attention_score_approx": attention_score_approx
        }

        return obs, reward, terminated, False, info

    def render(self):
        """Display current state on the frame with action labels and stats.

        Renders the current frame with:
        - Top-left: Current step and violation count
        - Left side: Phone duration, gaze duration, attention score
        - Bottom bar: Suggested action label (ALL CLEAR/MONITOR/VIOLATION)

        Note:
            All OpenCV display calls are wrapped in try/except to handle
            headless environments gracefully without crashing.
        """
        if self._current_frame is None:
            return

        frame = self._current_frame.copy()

        try:
            # Get tracker state
            tracker_state = get_tracker_state()
            phone_dur = tracker_state["phone_duration"]
            gaze_dur = tracker_state["gaze_duration"]
            attention_approx = phone_dur * 0.4 + gaze_dur * 0.3

            # Top-left: Step and violations
            cv2.putText(frame, f"Step: {self._step_count}  |  Violations: {self._violation_count}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Left side stacked stats
            cv2.putText(frame, f"Phone: {phone_dur:.1f}s",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Gaze: {gaze_dur:.1f}s",
                       (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Attention: {attention_approx:.2f}",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Bottom bar with action label (if obs exists)
            if self._last_obs is not None:
                # Determine most likely action based on current state
                if phone_dur > 3.0 or gaze_dur > 4.0:
                    action_label = "VIOLATION"
                    color = (0, 0, 255)  # Red
                elif phone_dur > 1.0 or gaze_dur > 1.5:
                    action_label = "MONITOR"
                    color = (0, 165, 255)  # Orange
                else:
                    action_label = "ALL CLEAR"
                    color = (0, 255, 0)  # Green

                # Semi-transparent bottom bar
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 430), (640, 480), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

                # Action label in large text
                cv2.putText(frame, action_label, (200, 465),
                           cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

            cv2.imshow("SmartRoad AI - Driver Environment", frame)
            cv2.waitKey(1)

        except cv2.error:
            # Headless environment or display error - silently skip
            pass

    def episode_stats(self):
        """Return summary statistics for the current/completed episode.

        Returns:
            dict: Episode statistics with keys:
                - total_steps: Number of steps taken in episode
                - total_violations_flagged: Number of times agent chose VIOLATION action
                - max_phone_duration_seen: Longest phone distraction observed (seconds)
                - max_gaze_duration_seen: Longest gaze away observed (seconds)
                - mean_reward: Average reward per step

        Note:
            Useful for analyzing agent performance and debugging reward shaping.
            Called after episode termination to log results.
        """
        mean_reward = self._total_reward / max(self._step_count, 1)

        return {
            "total_steps": self._step_count,
            "total_violations_flagged": self._total_violations_flagged,
            "max_phone_duration_seen": self._max_phone_duration_seen,
            "max_gaze_duration_seen": self._max_gaze_duration_seen,
            "mean_reward": mean_reward
        }


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    print("Testing DriverEnv...")

    # Test with FakePipeline
    env = DriverEnv(pipeline_fn=None, max_steps=100)

    print("Running check_env()...")
    check_env(env.unwrapped)
    print("check_env() passed!")

    # Test episode
    obs, info = env.reset(seed=42)
    print(f"Initial obs: {obs}")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, terminated={terminated}")

        if terminated:
            break

    stats = env.episode_stats()
    print(f"Episode stats: {stats}")

    print("DriverEnv tests passed!")
