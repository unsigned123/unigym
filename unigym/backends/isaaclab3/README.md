# backends/isaaclab3 — IsaacLab 3.x backend (EA skeleton)

状态：**EA 冒烟级占位**（v0.37.0，backend-traits §R M3'）。注册表条目存在
（`unigym.backends.registry`，`isaaclab3`），版本探测路由就绪，**能力 trait
集为空**——本包当前不实现任何能力 Protocol。

## 何时被选中

- config `env.backend: isaaclab3` 显式钉死（isaaclab 3.x 栈缺失时
  `select_backend` fail-fast `{error, hint}`）；
- `env.backend: isaaclab`（缺省/别名）在探测到 isaaclab >= 3.0 时自动路由
  到本后端——**生产默认档仍是 isaaclab2**，EA→GA 演进期间用显式
  `isaaclab2` 钉死既有 config。

## 缺席能力清单（对照 registry trait 表，全部未实现）

ENV_STEP_RESET / ACTION_FACES_NAMED / JOINT_PERMUTATION（env 容器三必备）、
CAMERAS_RGB / CAMERAS_DEPTH_RIG / DEPTH_OBS_LEGACY / RECORDER_VIDEO、
CONTACTS_GROUPS / CONTACTS_OVERRIDE、PROPS_RIGID / PROPS_ARTICULATION_URDF /
PROPS_ARTICULATION_USD / PROPS_LATCH / PROPS_ACTUATED_ACTIONS、
DIRECTOR_TELEPORT / DIRECTOR_SETVEL / DIRECTOR_ATTACH_WELD /
DIRECTOR_ATTACH_SLAVE / DIRECTOR_GRIPPER、TERRAIN / EVENTS_DR /
INIT_POSE / SPAWN_XY_YAW / MOCAP。

按 R.2 布局，实现落在 `backends/isaaclab3/`（env / cameras / contacts /
props / director_ops / terrains / actuators / dr_events），每落
一个能力即在 `registry.ISAACLAB3_TRAITS` 补 trait 声明并从本清单划掉。

## 3.0 破坏面的吸收原则（无 compat shim）

il3 与 il2 是两个平等后端：3.0 特有的约定（XYZW 数据视图、`write_*_index`
拆分、ProxyArray、接触力改名将 B4 等，设计 §2）在本包内**直接实现**，
仓库内部 wxyz 不变量经 `unigym.core.quat.from_isaaclab_data` /
`to_isaaclab_cfg` 在数据边界转换——不在任何共享层做按版本分支。

## EA 冒烟路径（栈就绪后）

容器内 `import unigym.backends.isaaclab3` + `select_backend("isaaclab3")`
探针 + `Isaaclab3Env` 构造应得到上述 stub NotImplementedError（fail-fast 面
自证）。
