
##  Actor(Policy) Model 行动者（策略）模型
##  Actor 行动者模型 将状态映射到动作

from keras import layers, models, optimizers
from keras import backend as K
import tensorflow as tf

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
  
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=400, activation='relu')(states)  # 32  TODO1 more hidden units
        net = layers.Dense(units=300, activation='relu')(net)     # 64
        #net = layers.Dense(units=32, activation='relu')(net)     # 32

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # 注意: 损失函数如何使用动作值（Q 值）梯度进行定义：
        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))   # this is a placeholder        
        loss = K.mean(-action_gradients * actions) 

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=0.0001)  # TODO2 Actor original lr=0.001, set as in paper
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()], 
            outputs=[],
            updates=updates_op)
        # 以上（动作值 Q值）梯度需要使用评论者模型计算，并在训练时提供梯度
        # 因此指定为在训练函数中使用的“输入”的一部分


## Critic (Value) Model 评论者模型
## Critic 评论者模型 将（状态、动作）对映射到它们的 Q 值
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
        # 将（状态、动作）对映射到它们的 Q 值
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=400, activation='relu')(states)      # 32  more units
        net_states = layers.Dense(units=300, activation='relu')(net_states)  # 64

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=400, activation='relu')(actions)    # 32
        net_actions = layers.Dense(units=300, activation='relu')(net_actions)# 64

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # 合并states, actions 单独的‘路径’（迷你子网络）
        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)  # default lr=0.001, same as paper
        self.model.compile(optimizer=optimizer, loss='mse')

        # 最终输出是任何给定（状态、动作）对的 Q 值
        # 计算此 Q 值相对于相应动作向量的梯度，以用于训练行动者模型 - 明确执行的一步
        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # 定义一个单独的函数来访问以上梯度
        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
 
