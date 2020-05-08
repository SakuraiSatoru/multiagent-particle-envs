import copy
import numpy as np
from multiagent.scenario import BaseScenario
from rl_drone_construction.utils.entities import PPODrone, TargetLandmark
from rl_drone_construction.utils.worlds import FacadeCoatingWorld


class Scenario(BaseScenario):
    def make_world(self):
        n_lidar_per_agent = 256
        world = FacadeCoatingWorld(n_lidar_per_agent=n_lidar_per_agent, mem_frames=3, dt=0.08)
        json_path = __file__[:-2]+'json'
        world.init_from_json(path=json_path)
        num_agents = len(world.docks)
        world.collaborative = False

        world.agents = [PPODrone(uid=i) for i in range(num_agents)]
        world.landmarks = [TargetLandmark() for i in range(num_agents)]

        for i, supply in enumerate(world.supplies):
            supply.size = 2

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.size = 0.3
            agent.lidar_range = 4
            agent.pseudo_collision_range = 0.4
            agent.construct_range = world.supplies[0].size - agent.size
            agent.construct_maxvel = 10
            agent.construct_capacity = 1
            agent.cur_construct_capacity = 0
            agent.battery = 100
            agent.waiting = 100 * i

        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.reset_progress()
        for i, agent in enumerate(world.agents):
            agent.load = None
            agent.cur_construct_capacity = 0
            agent.dock = world.docks[i]
            agent.state = copy.copy(world.docks[i].state)
            agent.target = world.targets[i]
            agent.target.state = copy.copy(world.docks[i].state)  # assign a state for first step
            agent.previous_state = copy.copy(agent.state)

        for agent in world.agents:
            agent.agents_lidar = world.lidar.get_ray_lidar(agent)
            agent.lidar_memory = [agent.agents_lidar, agent.agents_lidar]
            agent.supervised = False
            agent.supervise_fn = []
            agent.supervise_param = []
            agent.supervise(agent.supervise_charge, [world])
            agent.supervise(agent.supervise_wait, [])

    def reward(self, agent, world):
        return 0

    def observation(self, agent, world):
        out = [np.concatenate(agent.lidar_memory + [agent.agents_lidar]),
               agent.state.p_vel,
               agent.target.state.p_pos - agent.state.p_pos,
               ]
        return np.concatenate(out)

    def done(self, agent, world):
        return False