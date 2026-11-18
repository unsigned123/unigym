# Observations

The run config declares every observation source explicitly:

```yaml
observations:
  robot: [go2]
  contact: [feet]
  object: [box]
  camera: [ego]
  site: [gripper]
  command: true
  prev_action: true
```

Rules:

- `robot` is required and every name must be a declared articulation.
- `contact` names must exist in `contacts:`, `object` names in `props:` /
  `objects:`, `camera` names in `cameras:`, and `site` names in the scene XML.
- `command` / `prev_action` are run-global slots (default `false`).
- Cameras declared only for recording/video stay outside `observations.camera`
  and never render on the observation tick.

The environment owns one typed registry (`env.obs`). Adapters read views:

```python
env.obs.robot("go2").joint_pos
env.obs.contact("feet").any()
env.obs.object("box").pos_w
env.obs.camera("ego").rgb
env.obs.site("gripper").pos_w
env.obs.command.value
env.obs.prev_action.value
```

Each tick produces an immutable `ObsSnapshot` (`step.obs`) and a record mapping
(`step.record`). `ObsSnapshot.record()` emits the canonical artifact keys
(`articulation/...`, `contact/...`, `object/...`, `site/...`); cameras,
command/prev_action and bookkeeping never enter that artifact record.
`meta["observations"]` carries the schema snapshot.

There is no adapter request list, catalog, or raw cache: adapters implement
`compute_obs(obs: ObsSnapshot, env)` and read typed views directly.
