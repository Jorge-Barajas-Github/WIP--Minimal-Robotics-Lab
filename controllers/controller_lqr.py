import sys
import os
import math
import numpy as np
from scipy.linalg import solve_continuous_are

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from environments.pendulum_env import PendulumEnv

def compute_lqr_gain():
    '''
    Computes the LQR gain matrix K.

    The pendulum state is:
        x = [angle error
            angular velocity error]

    The linearized system has the form:
        x_dot = A x + B u
    where:
        A     = system dynamics matrix
        B     = input matrix
        u     = torque input

    For an upright pendulum linearized near the unstable upright equilibrium:
        angle_error_dot = velocity
        velocity_dot = (g / l) * angle_error + (1 / (m * l^2)) * torque

    Q = state cost matrix
        Q tells the LQR controller how much it should care about state error.
        Q[0, 0] = penalty on angle error
        Q[1, 1] = penalty on angular velocity error
        Larger values make the controller try harder to reduce that error.

    R = control cost matrix
        R tells the LQR controller how much it should care about using large torque.
        Larger R means smoother, weaker torque.
        Smaller R means more aggressive torque.
    P = solution to the continuous-time algebraic Riccati equation.
    K = LQR gain matrix
        K = R^-1 B^T P

    The final controller uses:
        torque = -Kx
    '''

    # Physics: 
    g = 9.81 # gravity
    l = 1    # length
    m = 1    # mass

    A = np.array([
        [0.0, 1.0],
        [g/l, 0.0]])
    B = np.array([
        [0.0],
        [1.0 / (m* l**2)]])
    #LQR Cost Matrices
    Q = np.array([
        [10.0, 0.0],
        [0.0,  1.0]])
    R = np.array([
        [1.0]])
    
    # Solve RICATTI: 
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @P
    return K

K = compute_lqr_gain()

def lqr_controller(state):
    
    '''
    LQR Controller = Linear Quadratic Regulator.

    LQR is an optimal feedback controller for systems that can be approximated as linear near an operating point.
    For the pendulum, the operating point is usually the upright position.
    
    torque = force-like rotational command sent to the motor/joint
    torque = -Kx  =  u
    '''    
    
    # Step 1: get current state from the environment
    angle = state["angle"]
    angle = math.atan2(math.sin(angle), math.cos(angle))

    velocity = state["velocity"]

    # Step 2: define the target upright state
    target_angle = 0
    target_velocity = 0

    # state errors
    angle_error = angle - target_angle
    velocity_error = velocity - target_velocity

    # state error vector
    x = np.array([
        [angle_error],
        [velocity_error]])

    # LQR gain matrix
    torque = -K @ x
    torque = float(torque[0,0])

    return torque


env = PendulumEnv(initial_angle=0.2)
env.run(controller=lqr_controller)