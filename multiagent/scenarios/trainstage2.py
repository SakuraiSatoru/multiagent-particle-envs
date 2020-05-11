import numpy as np
import colorlover as cl
from multiagent.scenario import BaseScenario
from rl_drone_construction.utils.entities import Drone, TargetLandmark, ThreatEntity
from rl_drone_construction.utils.worlds import DroneWorldRayLidar


class Scenario(BaseScenario):
    def make_world(self):
        n_lidar_per_agent = 256
        world = DroneWorldRayLidar(n_lidar_per_agent=n_lidar_per_agent,
                                   mem_frames=1, dt=0.08)  #todo mem_frames
        # set any world properties first
        num_agents = 10
        num_targets = num_agents
        num_threats = 3
        world.collaborative = False

        world.agents = [Drone(i) for i in range(num_agents)]
        world.landmarks = [TargetLandmark() for i in range(num_targets)] + \
                          [ThreatEntity(uid=i) for i in range(num_threats)]

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
        # make initial conditions
        self.reset_world(world)
        return world

    def generate_random_threat(self, threat, world):
        threat.state.p_pos = np.random.uniform(-7, 7, 2)
        threat.size = np.random.uniform(0.4, 0.6)

    def generate_random_pose(self, agent, world):
        pos = np.random.uniform(-7, +7, 2)
        agent.state.p_pos = pos
        dis = np.linalg.norm(pos)
        c = np.any([self.is_collision(agent, t) for t in world.threats])
        while (dis >7 or c):
            pos = np.random.uniform(-7, +7, 2)
            agent.state.p_pos = pos
            dis = np.linalg.norm(pos)
            c = np.any([self.is_collision(agent, t) for t in world.threats])

    def generate_random_goal(self, agent, world):
        goal_pos = np.random.uniform(-7, +7, 2)
        agent.target.state.p_pos = goal_pos
        dis_origin = np.linalg.norm(goal_pos)
        dis_goal = np.linalg.norm(agent.state.p_pos - goal_pos)
        c = np.any([self.is_collision(agent.target, t) for t in world.threats])
        while dis_origin > 7 or dis_goal > 8 or dis_goal < 6 or c:
            goal_pos = np.random.uniform(-7, +7, 2)
            agent.target.state.p_pos = goal_pos
            dis_origin = np.linalg.norm(goal_pos)
            dis_goal = np.linalg.norm(agent.state.p_pos - goal_pos)
            c = np.any([agent.size + t.size > np.linalg.norm(agent.target.state.p_pos-t.state.p_pos) for t in world.threats])

    def reset_world(self, world):
        # random properties for agents
        world.goals = []
        n = 10
        colors = np.array(cl.to_numeric(cl.scales[str(10)]['div']['RdYlBu']))/255
        for threat in world.threats:
            self.generate_random_threat(threat, world)
        for i, agent in enumerate(world.agents):
            agent.size = np.random.uniform(0.2, 0.3)
            agent.pseudo_collision_range = agent.size + 0.1
            agent.color = colors[i%n]
            agent.target.color = colors[i%n]
            # agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            if i % 2 == 0:
                self.generate_random_pose(agent, world)
                self.generate_random_goal(agent, world)
            else:
                agent.state.p_pos = np.copy(world.agents[i-1].target.state.p_pos)
                agent.target.state.p_pos = np.copy(world.agents[i-1].state.p_pos)
            world.goals.append(np.copy(agent.target.state.p_pos))

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
            # TODO threat collision
            for t in world.threats:
                if self.is_collision(agent, t):
                    print(agent.name, 'collided with', t.name)
                    reward_c -= 15
                    agent.terminate = True
                else:
                    reward_c += self.collision_reward(agent, t)
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(agent, a):
                    print(agent.name, 'collided with', a.name)
                    reward_c -= 15
                    agent.terminate = True
                else:
                    reward_c += self.collision_reward(agent, a)

        if d < agent.construct_range and (np.abs(agent.state.p_vel) < 0.2).all():
            print(agent.name, 'reached target')
            reward_g += 15
            # self.generate_random_goal(agent, world)
            if agent.uid % 2 == 0:
                goals = [np.copy(world.goals[agent.uid]), np.copy(world.goals[agent.uid+1])]
            else:
                goals = [np.copy(world.goals[agent.uid-1]), np.copy(world.goals[agent.uid])]
            if np.all(agent.target.state.p_pos == goals[0]):
                agent.target.state.p_pos = goals[1]
            else:
                agent.target.state.p_pos = goals[0]
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