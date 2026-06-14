import time
import mujoco
import mujoco.viewer


PENDULUM_XML = """
<mujoco model="inverted_pendulum">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6"
               ambient="0.3 0.3 0.3"
               specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <asset>
    <texture type="skybox"
             builtin="gradient"
             rgb1="0.4 0.6 0.8"
             rgb2="0 0 0"
             width="512"
             height="512"/>

    <texture type="2d"
             name="groundplane"
             builtin="checker"
             rgb1="0.25 0.35 0.45"
             rgb2="0.15 0.25 0.35"
             width="300"
             height="300"/>

    <material name="groundplane"
              texture="groundplane"
              texuniform="true"
              texrepeat="5 5"
              reflectance="0.15"/>
  </asset>

  <worldbody>
    <light pos="0 0 4"
           dir="0 0 -1"
           directional="true"/>

    <geom name="floor"
          type="plane"
          size="5 5 0.1"
          pos="0 0 -1.5"
          material="groundplane"/>

    <!-- angle = 0 means pendulum is upright -->
    <body name="pendulum" pos="0 0 0">
      <joint name="pin"
             type="hinge"
             axis="0 1 0"
             damping="0.02"/>

      <geom name="rod"
            type="capsule"
            size="0.025"
            fromto="0 0 0 0 0 1.0"
            rgba="0.75 0.75 0.75 1"
            mass="0.05"/>

      <geom name="bob"
            type="sphere"
            size="0.08"
            pos="0 0 1.0"
            rgba="0.9 0.1 0.1 1"
            mass="1.0"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="torque"
           joint="pin"
           gear="1"
           ctrllimited="true"
           ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


class PendulumEnv:
    def __init__(self, initial_angle=0.1):
        self.model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
        self.data = mujoco.MjData(self.model)
        self.initial_angle = initial_angle
        self._reset()

    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = self.initial_angle
        mujoco.mj_forward(self.model, self.data)

    def get_state(self):
        return {
            "angle": float(self.data.qpos[0]),
            "velocity": float(self.data.qvel[0]),
            "time": float(self.data.time),
        }

    def apply_control(self, torque):
        torque = max(-10.0, min(10.0, float(torque)))
        self.data.ctrl[0] = torque

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def run(self, controller=None):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.lookat[0] = 0.0
            viewer.cam.lookat[1] = 0.0
            viewer.cam.lookat[2] = 0.5
            viewer.cam.distance = 3.5
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 90

            while viewer.is_running():
                state = self.get_state()

                if controller is None:
                    torque = 0.0
                else:
                    torque = controller(state)

                self.apply_control(torque)
                self.step()
                viewer.sync()
                time.sleep(self.model.opt.timestep*0.5)