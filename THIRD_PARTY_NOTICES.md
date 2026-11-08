# Third-party notices

The public core retains adapted code and data whose upstream terms require
attribution. Files not listed here are original UniGym code or self-authored
test/example assets released under the repository MIT license.

## RoboCasa / robosuite

`unigym/props/providers/robocasa/` contains declaration tables and fixture
construction logic translated from RoboCasa (NVIDIA GEAR), which is released
under the MIT license. RoboCasa's own fixture assets are distributed under
CC BY 4.0; this repository does not include those asset files.

Mechanism semantics from robosuite (MIT), such as visual-twin resolution and
naming-prefix conventions, are reimplemented rather than copied.

License text: `LICENSES/robocasa-LICENSE`.

## Assets

The retained in-package URDFs (`unigym/props/door.urdf`,
`unigym/props/drawer.urdf`) and the example/test MJCF models are
self-authored, MIT-clean minimal models.
