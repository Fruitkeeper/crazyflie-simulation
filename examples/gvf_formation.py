import numpy as np
import jax.numpy as jnp

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim


# GVF Trajectory Classes
class GvfEllipse:
    def __init__(self, center, alpha, a, b):
        self.center = np.array(center)
        self.alpha = alpha
        self.a, self.b = a, b

        self.cosa, self.sina = np.cos(alpha), np.sin(alpha)
        self.R = np.array([[self.cosa, self.sina], [-self.sina, self.cosa]])

    def phi(self, p):
        """Scalar field: φ = 0 defines the ellipse"""
        w = self.center
        pel = (p[:2] - w) @ self.R
        return (pel[0] / self.a) ** 2 + (pel[1] / self.b) ** 2 - 1

    def grad_phi(self, p):
        """Gradient of φ (perpendicular to trajectory)"""
        w = self.center
        pel = (p[:2] - w) @ self.R
        grad_2d = 2 * pel / [self.a**2, self.b**2] @ self.R.T
        return np.array([grad_2d[0], grad_2d[1], 0.0])  # Add z component


# E matrix for 90-degree rotation (gradient -> tangent)
E = np.array([[0, 1, 0],
              [-1, 0, 0],
              [0, 0, 0]])


def get_formation_offsets(formation_type, num_agents, size=0.3):
    """Generate formation offsets in local frame"""
    offsets = []

    if formation_type == 'circle':
        for i in range(num_agents):
            angle = 2 * np.pi * i / num_agents
            # offset[0] = tangent direction, offset[1] = normal direction
            offsets.append([size * np.cos(angle), size * np.sin(angle)])

    elif formation_type == 'square':
        if num_agents == 4:
            offsets = [
                [size, size],    # Front-right
                [-size, size],   # Front-left
                [-size, -size],  # Back-left
                [size, -size],   # Back-right
            ]
        else:
            # Approximate square for other numbers
            side = int(np.sqrt(num_agents))
            for i in range(num_agents):
                row = i // side
                col = i % side
                offsets.append([
                    (col - side/2) * size,
                    (row - side/2) * size
                ])

    return np.array(offsets)


def compute_gvf_formation_control(positions, trajectory, formation_offsets,
                                   k_n=0.8, k_e=0.4, k_formation=5.0,
                                   direction='ccw', max_speed=0.5):
    """
    Parametric GVF with formation control

    Args:
        positions: (n_drones, 3) array of current positions
        trajectory: GVF trajectory object with phi() and grad_phi()
        formation_offsets: (n_drones, 2) offsets in [tangent, normal] frame
        k_n: Normal convergence gain
        k_e: Tangential circulation gain
        k_formation: Formation maintenance gain
        direction: 'ccw' or 'cw'
        max_speed: Maximum speed limit

    Returns:
        (n_drones, 13) state control commands
    """
    n_drones = len(positions)
    commands = np.zeros((n_drones, 13))

    # Compute centroid
    centroid = np.mean(positions, axis=0)

    # Get trajectory properties at centroid
    phi_centroid = trajectory.phi(centroid)
    grad_centroid = trajectory.grad_phi(centroid)
    grad_norm = np.linalg.norm(grad_centroid)

    if grad_norm > 1e-6:
        grad_normalized = grad_centroid / grad_norm
    else:
        grad_normalized = grad_centroid

    # Tangent and normal directions
    dir_sign = -1 if direction == 'ccw' else 1
    tangent_direction = dir_sign * (E @ grad_normalized)
    normal_direction = -grad_normalized

    # For each drone
    for i in range(n_drones):
        pos = positions[i]

        # Formation offset in world frame
        local_offset = formation_offsets[i]
        offset_tangent = local_offset[0] * tangent_direction
        offset_normal = local_offset[1] * normal_direction
        desired_pos = centroid + offset_tangent + offset_normal

        # GVF at agent position
        phi = trajectory.phi(pos)
        grad = trajectory.grad_phi(pos)
        grad_norm_agent = np.linalg.norm(grad)

        if grad_norm_agent > 1e-6:
            grad_normalized_agent = grad / grad_norm_agent
        else:
            grad_normalized_agent = grad

        # GVF control law
        u_normal = -k_n * phi * grad_normalized_agent
        u_tangent = dir_sign * k_e * (E @ grad_normalized_agent)
        u_formation = k_formation * (desired_pos - pos)

        # Combined velocity
        velocity = u_normal + u_tangent + u_formation

        # Speed limiting
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            velocity = velocity / speed * max_speed

        # State control: [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]
        commands[i, :3] = desired_pos  # Target position
        commands[i, 3:6] = velocity    # Target velocity
        commands[i, 9] = 0.0           # Yaw = 0

    return commands


def main():
    # Simulation parameters
    n_drones = 5
    duration = 20.0  # seconds
    fps = 30

    # Create simulator
    sim = Sim(
        n_worlds=1,
        n_drones=n_drones,
        physics=Physics.analytical,
        control=Control.state,
        freq=500,           # Physics at 500Hz
        state_freq=100,     # GVF controller at 100Hz
        attitude_freq=500,
        device="cpu",
    )

    sim.reset()

    # GVF trajectory (ellipse at z=1.0)
    trajectory = GvfEllipse(
        center=[0.0, 0.0, 1.0],
        alpha=0.0,      # No rotation
        a=1.5,          # Semi-major axis
        b=1.0           # Semi-minor axis
    )

    # Formation (circle formation)
    formation_offsets = get_formation_offsets('circle', n_drones, size=0.3)

    print(f"Running GVF formation control with {n_drones} drones")
    print(f"Trajectory: Ellipse (a={trajectory.a}, b={trajectory.b})")
    print(f"Formation: Circle (radius=0.3m)")
    print(f"Duration: {duration}s")
    print("\nPress 'q' in the render window to quit early\n")

    # Control loop
    control_steps = int(duration * sim.data.controls.state_freq)
    physics_steps_per_control = sim.freq // sim.data.controls.state_freq

    for i in range(control_steps):
        # Get current positions (world 0)
        positions = np.array(sim.data.states.pos[0])  # (n_drones, 3)

        # Compute GVF + formation control
        commands = compute_gvf_formation_control(
            positions,
            trajectory,
            formation_offsets,
            k_n=0.8,
            k_e=0.4,
            k_formation=5.0,
            direction='ccw',
            max_speed=0.5
        )

        # Send commands (need to add batch dimension for n_worlds)
        commands_jax = jnp.array(commands[None, :, :])  # (1, n_drones, 13)
        sim.state_control(commands_jax)

        # Step physics
        sim.step(physics_steps_per_control)

        # Render at target fps
        if (i * fps) % sim.data.controls.state_freq < fps:
            sim.render()

    sim.close()
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
