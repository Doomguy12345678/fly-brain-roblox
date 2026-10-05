# fly-brain-roblox

A screen-only Python agent scaffold for Roblox anomaly-observation games. It
captures pixels from a focused game window, compares them to stored room
references, and trains a Stable-Baselines3 PPO policy through Gymnasium.

## Architecture

- `screen_capture.py` captures a desktop or configured region at a controlled
  rate and resizes frames for the agent.
- `capture_phone.py` observes a phone screen shown in a desktop mirroring
  window, saving sampled screenshots and visual-change summaries without
  sending input to the phone.
- `anomaly_detector.py` calculates frame similarity, thresholded difference
  maps, template matches, and approximate changed-region events.
- `memory_manager.py` stores normal-room reference images and an append-only
  observation history.
- `vision_processor.py` forms policy observations from the current RGB frame,
  a difference map, and a stack of recent frames.
- `roblox_env.py` maps discrete camera, interact, report, door, and wait actions
  to keyboard inputs. The Roblox window must be focused before training.
- `feature_extractor.py` uses a small CNN to encode the visual observations.
- `train.py` trains PPO and writes checkpoints, TensorBoard logs, screenshots,
  and per-step replay records.
- `evaluate.py` runs a saved policy and prints episode rewards and report
  outcomes.
- `config.py` contains the typed capture, vision, reward, input, and path
  settings.

This project does not read Roblox internals, game memory, or network traffic.
Visual changed regions are not semantic object labels; moving the camera,
lighting changes, or animation may also look anomalous. Use stable room views
and tune the thresholds for each game.

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run screen capture in a graphical desktop session. For the existing PC
keyboard-control mode, focus the Roblox window and enter a normal room before
starting: the first environment reset saves that screen as the scene's
baseline. To explicitly replace a baseline, remove that scene's image under
`memory/references/` before starting again.

The default capture region is the primary monitor. Restrict capture to a
game-window region with `--roi LEFT TOP WIDTH HEIGHT` (screen coordinates).
The game window should remain in that region throughout training.

## Observe Roblox on an iPhone

On Windows, first connect the iPhone and open a mirroring application that
shows its live display in a desktop window. The Python tool captures pixels
from that visible window; it does not connect to iOS over USB itself, launch
Roblox, tap the phone, or control the mirrored app. You keep the phone's touch
controls manual. Mirroring applications differ in whether they support iPhone
USB mirroring; AirDroid Cast documents phone control through its desktop client,
but iPhone/Roblox touch forwarding must be confirmed on your own setup. See the
[AirDroid Cast product page](https://www.airdroid.com/cast/).

With the mirrored Roblox game showing a normal room, capture the desktop
coordinates of the mirror window and run:

```bash
python capture_phone.py \
  --scene anomaly-detect-phone-room \
  --roi 100 80 430 760 \
  --fps 5
```

The first frame becomes the saved normal-room reference if one does not exist.
The tool writes a screenshot every five frames by default, and logs a
similarity/change summary for every frame. Use `--replace-baseline` only when
the first visible frame is a known normal room and you intend to replace that
scene's existing baseline. `--duration 300` stops after five minutes; otherwise
press Ctrl+C to stop. `--archive-every 1` saves every frame.

This iPhone mode is observation/data collection only; it does not train PPO,
because the agent cannot perform phone actions or know the result of manually
performed taps. PPO training and evaluation described below remain for the
keyboard-controlled desktop Roblox client.

## Report feedback and rewards

Successful and unsuccessful reports must be identified from visible game
feedback. Capture small image templates from each game's success and failure
indicators and pass them to training/evaluation. Templates can be matched
anywhere within the captured region. Because captured frames are resized to
160x120 by default, template crops must be at that same pixel scale and fit
inside the resized frame.

```bash
python train.py \
  --scene anomaly-detect-room \
  --roi 100 80 1280 720 \
  --success-template templates/anomaly-detect/success.png \
  --failure-template templates/anomaly-detect/failure.png \
  --timesteps 100000
```

If the game has no visible outcome indicator, `--neutral-without-feedback`
explicitly opts into neutral rewards for unrecognized reports. That mode cannot
teach PPO which reports were correct. Use separate scene names for distinct
normal-room baselines. Override controls for games with different key layouts
by repeating `--key ACTION=KEY` in both training and evaluation, for example
`--key look_left=a --key look_right=d --key report_anomaly=f`.

Run evaluation with matching feedback templates and capture settings:

```bash
python evaluate.py \
  --scene anomaly-detect-room \
  --model models/anomaly-detect-room_ppo_final.zip \
  --roi 100 80 1280 720 \
  --success-template templates/anomaly-detect/success.png \
  --failure-template templates/anomaly-detect/failure.png
```

Training checkpoints are written to `models/`, TensorBoard data and Gymnasium
episode statistics to `logs/`, archived frames to `screenshots/`, references
and observation history to `memory/`, and action/reward replay records to
`replays/`.

## Python integration

The environment can also be configured directly for another game's controls:

```python
from pathlib import Path

from config import AgentConfig, CaptureConfig, ProjectPaths, RewardConfig, VisionConfig
from roblox_env import RobloxAnomalyEnv

controls = AgentConfig(
    key_bindings={
        "look_left": "a",
        "look_right": "d",
        "look_up": "w",
        "look_down": "s",
        "interact": "e",
        "report_anomaly": "r",
        "open_door": "e",
        "close_door": "q",
    }
)
env = RobloxAnomalyEnv(
    scene_name="keep-door-locked-room",
    capture_config=CaptureConfig(fps=5),
    vision_config=VisionConfig(),
    agent_config=controls,
    reward_config=RewardConfig(
        success_templates=(Path("templates/door-game/success.png"),),
        failure_templates=(Path("templates/door-game/failure.png"),),
    ),
    paths=ProjectPaths(),
)
```

The Gymnasium observation is a dictionary of `uint8` arrays (`frame`,
`difference`, and `history`); the discrete action space is defined by
`GameAction` in `roblox_env.py`.
