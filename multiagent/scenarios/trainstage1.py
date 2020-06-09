import numpy as np
import colorlover as cl
from multiagent.scenario import BaseScenario
from mdac.utils.entities import Drone, TargetLandmark, SupplyEntity
from mdac.utils.worlds import DroneWorld


class Scenario(BaseScenario):
    def make_world(self):
        n_lidar_per_agent = 256
        world = DroneWorld(n_lidar_per_agent=n_lidar_per_agent,
                           mem_frames=1, dt=0.08)
        num_agents = 5
        num_targets = num_agents
        world.collaborative = False

        world.agents = [Drone(uid=i) for i in range(num_agents)]
        world.landmarks = [TargetLandmark() for i in range(num_targets)]

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.lidar_range = 4.0
            agent.target = world.landmarks[i]
            agent.construct_range = 0.1
        for i, landmark in enumerate(world.landmarks):
            landmark.collide = False
            landmark.movable = False
            if isinstance(landmark, TargetLandmark):
                landmark.name = 'landmark %d' % i
                landmark.size = 0.05
            if isinstance(landmark, SupplyEntity):
                landmark.name = 'supply %d' % i
                landmark.size = 1.5
        # make initial conditions
        self.reset_world(world)
        return world

    def generate_random_pose(self, agent):
        pos = np.random.uniform(-7, +7, 2)
        dis = np.linalg.norm(pos)
        while (dis > 7):
            pos = np.random.uniform(-7, +7, 2)
            dis = np.linalg.norm(pos)
        agent.state.p_pos = pos

    def generate_random_goal(self, agent):
        goal_pos = np.random.uniform(-7, +7, 2)
        dis_origin = np.linalg.norm(goal_pos)
        dis_goal = np.linalg.norm(agent.state.p_pos - goal_pos)
        while (dis_origin > 7 or dis_goal > 8 or dis_goal < 6):
            goal_pos = np.random.uniform(-7, +7, 2)
            dis_origin = np.linalg.norm(goal_pos)
            dis_goal = np.linalg.norm(agent.state.p_pos - goal_pos)
        agent.target.state.p_pos = goal_pos


    def reset_world(self, world):
        colors = np.array(cl.to_numeric(cl.scales['5']['div']['RdYlBu']))/255
        for i, agent in enumerate(world.agents):
            agent.size = np.random.uniform(0.2, 0.3)
            agent.pseudo_collision_range = agent.size + 0.1
            agent.color = colors[i%5]
            agent.target.color = colors[i%5]
            self.generate_random_pose(agent)
            self.generate_random_goal(agent)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.previous_state.p_pos = np.copy(agent.state.p_pos)
            agent.previous_state.p_vel = np.copy(agent.state.p_vel)
            agent.state.c = np.zeros(world.dim_c)
            agent.terminate = False
        for agent in world.agents:
            agent.agents_lidar = world.lidar.get_ray_lidar(agent)
            agent.lidar_memory = [agent.agents_lidar, agent.agents_lidar]

    def is_collision(self, agent1, agent2):
        dist = np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos)
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def collision_reward(self, agent, entity):
        if agent is entity:
            return 0
        if agent.pseudo_collision_range is not None:
            p_range = agent.pseudo_collision_range
        else:
            p_range = agent.size
        d = np.linalg.norm(agent.state.p_pos-entity.state.p_pos)
        s = agent.size + entity.size
        if d > p_range + entity.size:
            return 0
        if d <= s:
            return -15
        return ((d - s) / (p_range - agent.size) - 1) * 15


    def reward(self, agent, world):
        prev_d = np.linalg.norm(agent.previous_state.p_pos - agent.target.state.p_pos)
        d = np.linalg.norm(agent.state.p_pos - agent.target.state.p_pos)
        reward_g = (prev_d - d) * 2.5
        reward_c = 0

        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(agent, a):
                    print(agent.name, 'collided')
                    reward_c -= 15
                    agent.terminate = True
                else:
                    reward_c += self.collision_reward(agent, a)

        if d < agent.construct_range and (np.abs(agent.state.p_vel) < 0.2).all():
            print(agent.name, 'reached target')
            reward_g += 15
            # agent.target.state.p_pos = np.random.uniform(-6, +6, world.dim_p)
            self.generate_random_goal(agent)
            agent.terminate = True
        return reward_c + reward_g



    def observation(self, agent, world):
        out = [np.concatenate(agent.lidar_memory + [agent.agents_lidar]),
               agent.state.p_vel,
               agent.target.state.p_pos - agent.state.p_pos,
               ]
        return np.concatenate(out)

    def done(self, agent, world):
        return agent.terminate