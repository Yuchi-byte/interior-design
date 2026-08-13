"""Warmup restarts: several draws audition for the noisiest part of the walk."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WarmupRestarts:
    """How many trajectories are started, and how far, before one is kept.

    FMRG's warmup particle selection (arXiv:2604.27147, §E.5): K independent
    draws each walk the first part of the guidance window, the reward is
    evaluated on each one's endpoint estimate, and the best continues alone.
    Upstream reports K=3 as sufficient and roomgen's own tuning found the same —
    it is the knob that rescues a cell where the single trajectory never took,
    and it does nothing for a cell that was already working.

    It is a greedy selection *on the reward being steered by*, which is worth
    saying plainly: on a shell-only depth energy the global optimum is an
    unfurnished room, so the selection prefers emptiness exactly as the guidance
    does. It buys robustness, not honesty; that is the preference term's job.

    Args:
        particles: how many draws audition, at the cost of `particles - 1`
            partial walks. 1 is the single trajectory, bit for bit — the winner
            of a field of one is not scored at all.
        steps: how many steps each draw walks before the field is judged, in
            walk steps rather than a fraction, since the guidance window is
            already counted that way.
    """

    particles: int
    steps: int

    def __post_init__(self) -> None:
        if self.particles < 1:
            raise ValueError(f"particles must be at least 1, got {self.particles}")
        if self.steps < 1:
            raise ValueError(f"steps must be at least 1, got {self.steps}")

    def check(self, total: int) -> None:
        """Reject a warmup that would consume the whole walk it is warming up."""
        if self.steps >= total:
            raise ValueError(
                f"a warmup of {self.steps} steps leaves nothing of a {total}-step "
                f"walk to continue: the particles would be sampled to the end and "
                f"the selection would be a best-of-{self.particles} over finished "
                f"images, which is a different method"
            )
