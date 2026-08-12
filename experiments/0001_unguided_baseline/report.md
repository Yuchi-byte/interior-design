# 0001_unguided_baseline

**Not implemented, and superseded before it was.** There is no `run.py`, no
`conf/` and no run: this directory is the number and this note.

## Why it is empty

0001 was to establish, in runs of its own, the two things a guidance framework
has to be read against — the VAE-reconstruction floor a perfect
geometry-preserving renovation still pays, and an unguided drift number to
compare against. `0002_geometric_guidance` measures both **in process**
instead:

- every guided sample is paired with an unguided control at the same photo,
  prompt, seed, sampler budget and trajectory, so the control is a measurement
  of that run rather than a remembered number from another one;
- each scene records its reconstruction floor from the photo round-tripped
  through the VAE alone.

Both are strictly better than a stored baseline, which leaves 0001 with nothing
to deliver. Its other intended deliverable — the two constants of a pass/fail
gate — has no home either: a framework reports metrics and the owner judges
them (`experiments/README.md`).

The unguided arm's numbers therefore live in 0002's `results.json` and report,
per (scene, seed), as `E_control`.
