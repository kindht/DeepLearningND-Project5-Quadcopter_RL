import numpy as np
from agents.helper import ReplayBuffer, OUNoise
from agents.DDPG_model import Actor, Critic

### DDPG: 智能体  
class DDPG_Agent():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        ###### TODO1 DDPG Actor/Critic Model  #######
        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0  #  initi default [0,0,0,0] ->ndarray
        self.exploration_theta = 0.15   # init default 0.15
        self.exploration_sigma = 0.2    # init default 0.3  -- TODO1.1 0.2 as in paper
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 1000000  # original = 100000 -- TODO1.2 10^6 as in paper
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor, same as paper
        self.tau = 0.001  # for soft update of target parameters -- TODO1.3 original 0.01
        
        self.all_rewards = []
        ###### END OF TODO1 DDPG Actor/Critic Model  #######
      

    def reset_episode(self):  # TODO2 Add noise/last_state
        self.noise.reset()
        
        self.episode_reward = 0.0
        self.count = 0
        state = self.task.reset()
        
        self.last_state = state
        return state
    
    def act(self, state):  #  TODO3 no change
        """Choose actions based on given state(s) as per current policy"""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]  # shape (1, 4)-> action.shape (4, )
        return list(action + self.noise.sample()) # add some noise for exploration
    
    def step(self, action, reward, next_state, done):  # TODO4 save experiences/learn
        '''count the total rewards in one episode'''
        self.episode_reward += reward
        self.count += 1
        
        # Save experience & reward etc.
        self.memory.add(self.last_state, action, reward, next_state, done)
        
        # Learn if enough sample are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample() # experiences <class 'list'>
            self.learn(experiences)
        
        # Roll over last state and action !!
        self.last_state = next_state
        
        '''Each Episode: average reward/total rewards'''
        if done:
            self.score = self.episode_reward / float(self.count) if self.count else 0.0
            self.all_rewards.append((self.score, self.episode_reward))
    
    def learn(self, experiences):  
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        
        # Compute Q targets for current states
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)  # if done, only rewards
        
        # Train critic model (local) given the targets
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local) using the sampled gradient
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  #  custom training function in Actor

        # Soft-update target models
        self.soft_update(self.actor_local.model, self.actor_target.model)
        self.soft_update(self.critic_local.model, self.critic_target.model)
   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        