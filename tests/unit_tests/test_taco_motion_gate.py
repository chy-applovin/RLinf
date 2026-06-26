"""Unit tests for the TACO tracking reward's object motion gate.

These avoid building a real MuJoCo ``TacoEnv``: the onset detection is a pure
function of the demo qpos array, and ``TrackingReward.compute`` only touches a
handful of sub-env attributes, which we stub out. Run with::

    pytest tests/unit_tests/test_taco_motion_gate.py
"""

import numpy as np

from rlinf.envs.taco.rewards import TrackingReward, _object_motion_onset

# qpos layout for the stub: 2 hand dofs, then two 7-dof free joints
# (3 translation + 4 quaternion) for tool and target.
HAND_DIM = 2
TOOL_SLICE = slice(2, 9)
TARGET_SLICE = slice(9, 16)
NQ = 16


def _make_demo(num_frames: int, tool_onset: int, target_onset: int) -> np.ndarray:
    """Demo where tool/target step 0.05 m away from rest at the given frames."""
    demo = np.zeros((num_frames, NQ), dtype=np.float64)
    # identity-ish quaternions so the layout is realistic (unused by the reward)
    demo[:, TOOL_SLICE.start + 3] = 1.0
    demo[:, TARGET_SLICE.start + 3] = 1.0
    for t in range(num_frames):
        if t >= tool_onset:
            demo[t, TOOL_SLICE.start] = 0.05
        if t >= target_onset:
            demo[t, TARGET_SLICE.start] = 0.05
    return demo


# --------------------------------------------------------------- onset helper
def test_onset_detects_first_moving_frame():
    demo = _make_demo(num_frames=6, tool_onset=3, target_onset=5)
    assert _object_motion_onset(demo, TOOL_SLICE, 0.01) == 3
    assert _object_motion_onset(demo, TARGET_SLICE, 0.01) == 5


def test_onset_never_moves_returns_num_frames():
    demo = _make_demo(num_frames=6, tool_onset=3, target_onset=99)
    # target never crosses threshold -> "never moves" sentinel == num_frames
    assert _object_motion_onset(demo, TARGET_SLICE, 0.01) == 6


def test_onset_threshold_boundary():
    demo = _make_demo(num_frames=4, tool_onset=2, target_onset=99)
    # the 0.05 m step is above 0.01 m but below a 0.1 m threshold
    assert _object_motion_onset(demo, TOOL_SLICE, 0.01) == 2
    assert _object_motion_onset(demo, TOOL_SLICE, 0.1) == 4


# ------------------------------------------------------------ compute() gating
class _StubSpec:
    tool_obj_qpos = TOOL_SLICE
    target_obj_qpos = TARGET_SLICE
    hand_dim = HAND_DIM


class _StubEpisode:
    name = "stub_ep"
    spec = _StubSpec()

    def __init__(self, demo):
        self.qpos_demo = demo


class _StubData:
    def __init__(self, qpos):
        self.qpos = qpos


class _StubSub:
    """Minimal stand-in exposing only what TrackingReward.compute reads."""

    def __init__(self, demo, step):
        self.episode = _StubEpisode(demo)
        self.steps = step
        self.start_frame = 0
        # perfect tracking: sim qpos == demo qpos at the aligned frame
        self.data = _StubData(demo[step].copy())

    def demo_qpos(self, executed_steps):
        t = min(self.start_frame + executed_steps, self.episode.qpos_demo.shape[0] - 1)
        return self.episode.qpos_demo[t]

    @property
    def demo_frame(self):
        return min(
            self.start_frame + self.steps, self.episode.qpos_demo.shape[0] - 1
        )


def _reward_cfg(gate: bool):
    return {
        "type": "tracking",
        "tool_pos_weight": 2.0,
        "target_pos_weight": 0.5,
        "hand_qpos_weight": 0.1,
        "tool_pos_scale": 0.05,
        "target_pos_scale": 0.05,
        "hand_qpos_scale": 0.5,
        "contact_weight": 0.0,
        "obj_motion_gate": gate,
        "obj_motion_threshold_m": 0.01,
    }


def test_gate_zeros_object_terms_before_onset():
    demo = _make_demo(num_frames=6, tool_onset=3, target_onset=5)
    rf = TrackingReward(_reward_cfg(gate=True))

    # Before either onset: both object gates off, reward == hand term only.
    sub = _StubSub(demo, step=1)
    r, info = rf.compute(sub)
    assert info["tool_gate"] == 0.0 and info["target_gate"] == 0.0
    # perfect tracking -> exp(0)=1 for the hand term: r == w_hand / W
    assert np.isclose(r, rf.w_hand / rf.norm)


def test_gate_turns_on_at_onset_per_object():
    demo = _make_demo(num_frames=6, tool_onset=3, target_onset=5)
    rf = TrackingReward(_reward_cfg(gate=True))

    # Frame 3: tool on, target still off.
    sub = _StubSub(demo, step=3)
    _, info = rf.compute(sub)
    assert info["tool_gate"] == 1.0 and info["target_gate"] == 0.0

    # Frame 5: both on.
    sub = _StubSub(demo, step=5)
    _, info = rf.compute(sub)
    assert info["tool_gate"] == 1.0 and info["target_gate"] == 1.0


def test_gate_disabled_matches_ungated_and_keeps_full_weight():
    demo = _make_demo(num_frames=6, tool_onset=3, target_onset=5)
    rf_off = TrackingReward(_reward_cfg(gate=False))
    rf_on = TrackingReward(_reward_cfg(gate=True))

    # At a fully-active frame the gated reward equals the ungated reward
    # (W unchanged -> active terms identical).
    sub_off = _StubSub(demo, step=5)
    sub_on = _StubSub(demo, step=5)
    r_off, info_off = rf_off.compute(sub_off)
    r_on, _ = rf_on.compute(sub_on)
    assert np.isclose(r_off, r_on)
    # gate disabled => gates report 1.0 (no gating applied)
    assert info_off["tool_gate"] == 1.0 and info_off["target_gate"] == 1.0
    # perfect tracking everywhere -> ungated per-step reward is exactly 1.0
    assert np.isclose(r_off, 1.0)
