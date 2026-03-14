import numpy as np
from rl_environment import DriverEnv

def stress_test():
    """
    Runs comprehensive stress test on DriverEnv with FakePipeline.

    Tests:
    - 5 full episodes with max_steps=300
    - Observation bounds validation
    - No NaN or inf values
    - Phone duration never exceeds 30.0
    """
    print("=" * 60)
    print("STARTING STRESS TEST")
    print("=" * 60)

    env = DriverEnv(pipeline_fn=None, max_steps=300)

    all_obs_values = []

    for episode in range(1, 6):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode}")
        print('='*60)

        obs, info = env.reset(seed=100 + episode)
        terminated = False
        step = 0

        while not terminated:
            step += 1

            # Random action
            action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Collect obs for validation
            all_obs_values.append(obs.copy())

            # Print every 50 steps
            if step % 50 == 0:
                print(f"\nStep {step}:")
                print(f"  Episode: {episode}")
                print(f"  Obs: {obs}")
                print(f"  Action: {action} ({'ALL_CLEAR' if action==0 else 'MONITOR' if action==1 else 'VIOLATION'})")
                print(f"  Reward: {reward:.2f}")
                print(f"  Attention Score: {info['attention_score_approx']:.2f}")
                print(f"  Phone Duration: {info['phone_duration']:.2f}s")
                print(f"  Gaze Duration: {info['gaze_duration']:.2f}s")

            # Check termination
            if terminated:
                stats = env.episode_stats()
                print(f"\n{'*'*60}")
                print(f"EPISODE {episode} COMPLETED")
                print('*'*60)
                print(f"Episode Stats:")
                print(f"  Total Steps: {stats['total_steps']}")
                print(f"  Total Violations Flagged: {stats['total_violations_flagged']}")
                print(f"  Max Phone Duration: {stats['max_phone_duration_seen']:.2f}s")
                print(f"  Max Gaze Duration: {stats['max_gaze_duration_seen']:.2f}s")
                print(f"  Mean Reward: {stats['mean_reward']:.2f}")
                break

    # Validation checks
    print(f"\n{'='*60}")
    print("VALIDATION CHECKS")
    print('='*60)

    all_obs_array = np.array(all_obs_values)

    # Check 1: No NaN or inf values
    has_nan = np.any(np.isnan(all_obs_array))
    has_inf = np.any(np.isinf(all_obs_array))

    print(f"[OK] No NaN values: {not has_nan}")
    print(f"[OK] No inf values: {not has_inf}")

    assert not has_nan, "FAILED: Found NaN values in observations"
    assert not has_inf, "FAILED: Found inf values in observations"

    # Check 2: Observation bounds
    # Indices 0-4 are binary or confidence (0-1)
    # Indices 5-7 are durations (0-30)
    # Indices 8-9 are binary (0-1)

    for i in range(5):  # Binary/confidence values
        min_val = all_obs_array[:, i].min()
        max_val = all_obs_array[:, i].max()
        print(f"[OK] Obs[{i}] range: [{min_val:.2f}, {max_val:.2f}] (expected [0.0, 1.0])")
        assert min_val >= 0.0 and max_val <= 1.0, f"FAILED: Obs[{i}] out of bounds"

    for i in range(5, 8):  # Duration values
        min_val = all_obs_array[:, i].min()
        max_val = all_obs_array[:, i].max()
        print(f"[OK] Obs[{i}] range: [{min_val:.2f}, {max_val:.2f}] (expected [0.0, 30.0])")
        assert min_val >= 0.0 and max_val <= 30.0, f"FAILED: Obs[{i}] out of bounds"

    for i in range(8, 10):  # Binary values
        min_val = all_obs_array[:, i].min()
        max_val = all_obs_array[:, i].max()
        print(f"[OK] Obs[{i}] range: [{min_val:.2f}, {max_val:.2f}] (expected [0.0, 1.0])")
        assert min_val >= 0.0 and max_val <= 1.0, f"FAILED: Obs[{i}] out of bounds"

    # Check 3: Phone duration specifically never exceeds 30.0
    max_phone_duration = all_obs_array[:, 5].max()
    print(f"[OK] Max phone duration across all episodes: {max_phone_duration:.2f}s (must be <= 30.0)")
    assert max_phone_duration <= 30.0, f"FAILED: Phone duration exceeded 30.0s"

    # Check 4: Gaze duration never exceeds 30.0
    max_gaze_duration = all_obs_array[:, 6].max()
    print(f"[OK] Max gaze duration across all episodes: {max_gaze_duration:.2f}s (must be <= 30.0)")
    assert max_gaze_duration <= 30.0, f"FAILED: Gaze duration exceeded 30.0s"

    print(f"\n{'='*60}")
    print("STRESS TEST PASSED")
    print('='*60)
    print(f"Total observations validated: {len(all_obs_values)}")
    print("All assertions passed successfully!")


if __name__ == "__main__":
    stress_test()
