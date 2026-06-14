import sys
import os
import math

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from environments.pendulum_env import PendulumEnv


def pd_controller(state):
    
    '''
    PD Controller = type of feedback controller that works by minimizing error between a desired setpoint and the actual process variable.
    It merges proportional action P and derivative action D.
    P & D are not variables but are the two controller actions: 
        P = proportional action, based on position error
        D = derivative action, based on how fast the error is changing

    torque = force-like rotational command sent to the motor/joint = u
    u      = Kp*e + Kd*e_dot

    e      = position error
    e_dot  = derivative of the error, usually related to velocity

    Kp     = proportional gain
    Kd     = derivative gain
    '''    
    
    angle = state["angle"]
    angle = math.atan2(math.sin(angle), math.cos(angle))
    velocity = state["velocity"]

    target_angle = 0.0

    kp = 20.0
    kd = 0.5

    angle_error = target_angle - angle
    velocity_error = 0 - velocity

    # Step 3: combine them into torque
    torque = (kp * angle_error + kd * velocity_error)

    return torque


env = PendulumEnv(initial_angle=0.2)
env.run(controller=pd_controller)