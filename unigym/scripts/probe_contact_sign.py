"""Pin the MuJoCo mj_contactForce sign convention (contact-manager C3).

Sphere (mass 1 kg, gravity -9.81) settles on a plane. At rest the contact
force ON THE SPHERE must be +9.81 N along world +z. We print geom1/geom2,
their body ids, the raw contact-frame force, and the world-rotated force so
the backend convention (force on geom1 vs geom2 side) is measured, not guessed.
"""
import numpy as np
import mujoco

XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="ball" pos="0 0 1.05">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.05" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
for _ in range(1500):
    mujoco.mj_step(m, d)
mujoco.mj_forward(m, d)
print("ncon =", d.ncon)
for i in range(d.ncon):
    con = d.contact[i]
    f6 = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, f6)
    frame = con.frame.reshape(3, 3)
    f_world = frame.T @ f6[:3]
    g1b = m.geom_bodyid[con.geom1]
    g2b = m.geom_bodyid[con.geom2]
    mass = float(m.body('ball').mass)
    print(f"contact {i}: geom1={m.geom(con.geom1).name}(body={m.body(g1b).name}) "
          f"geom2={m.geom(con.geom2).name}(body={m.body(g2b).name})")
    print(f"  frc6[:3] (contact frame) = {f6[:3]}")
    print(f"  world force (frame.T @ f) = {f_world}")
    print(f"  ball mass = {mass}, expected on-ball = +mg = {mass * 9.81:.3f} N up")
    # measured convention: the returned force acts on the geom2 body;
    # the geom1 body receives the negation.
    assert m.body(g2b).name == "ball"
    assert np.allclose(f_world, [0.0, 0.0, mass * 9.81], atol=1e-2), f_world
    print("CONVENTION CONFIRMED: mj_contactForce -> force on GEOM2 body")
mujoco.mj_step(m, d)  # keep a live context (nothing else needed)
