# Thanks to : SoyGema, DavSuCar, davidleejy
# Visit https://github.com/SoyGema/Startcraft_pysc2_minigames for more info
# Feel free to PR your own agent!


import numpy as np
import sys
import random
import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, TimeDistributed, LSTM, Reshape, Dropout
from keras.optimizers import Adam, Adamax, Nadam
from keras.backend import set_image_dim_ordering
from absl import flags
 
from pysc2.env import sc2_env, environment
from pysc2.lib import actions
from pysc2.lib import features
 
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SARSAAgent
from rl.callbacks import Callback, FileLogger


# Actions from pySC2 API
 
FUNCTIONS = actions.FUNCTIONS
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_NO_OP = FUNCTIONS.no_op.id
_MOVE_SCREEN = FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = FUNCTIONS.Attack_screen.id
_SELECT_ARMY = FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

# Size of the screen and length of the window
 
_SIZE = 64
_WINDOW_LENGTH = 1
 
# Load and save weights for training
 
LOAD_MODEL = True  # True if the training process is already created
SAVE_MODEL = True
 
# global variable
 
episode_reward = 0
observation_cur = None
 
# Configure Flags for executing model from console
 
FLAGS = flags.FLAGS
flags.DEFINE_string("mini-game", "HalucinIce", "Name of the minigame")
flags.DEFINE_string("algorithm", "deepq", "RL algorithm to use")
flags.DEFINE_integer("screen_size", "64", "Resolution for screen actions.")
flags.DEFINE_integer("minimap_size", "32", "Resolution for minimap actions.")
flags.DEFINE_bool("verbose", False, "Intended to help a programmer read actions taken in real-time.")
flags.DEFINE_float("pause", 0.0, "Seconds to pause between consecutive actions. Intended to help a programmer observe the effect of an action taken in real-time.")
flags.DEFINE_bool("visualize", True, "Visualize game")
flags.DEFINE_integer("steps_per_ep", 150, "steps per episode.")
FLAGS(sys.argv)
 
 
# Processor
 
class SC2Proc(Processor):
    def process_observation(self, observation):
        """Process the observation as obtained from the environment for use an agent and returns it"""
        obs = observation[0].observation["feature_screen"][_PLAYER_RELATIVE]
        return np.expand_dims(obs, axis=2)
 
    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it"""
        batch = np.swapaxes(batch, 0, 1)
        return batch[0]
 
 
 
def args_random(actfunc):
    # E.g. of actfunc: pysc2.lib.actions.FUNCTIONS[81]
    args_given = []
    for arg in actfunc.args:
        arg_values = []
        for size in arg.sizes:
            if size == 0:
                arg_values.append(0)
            else:
                arg_values.append(np.random.randint(0, size))
        args_given.append(arg_values)
    return args_given
 
 
 
 
#  Define the environment
 
 
class Environment(sc2_env.SC2Env):
    """Starcraft II environmet. Implementation details in lib/features.py"""
 
    def step(self, action):
        """Apply actions, step the world forward, and return observations"""
        global episode_reward
        global observation_cur
        if observation_cur is None:
            action = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        else:
            function_id = np.random.choice(observation_cur.available_actions)
            args_chosen = args_random(actions.FUNCTIONS[function_id])
            action = actions.FunctionCall(function_id, args_chosen)
        obs = super(Environment, self).step([action])
        observation_cur = obs[0].observation
        observation = obs
        r = obs[0].reward
        done = obs[0].step_type == environment.StepType.LAST
        episode_reward += r
 
        return observation, r, done, {}
 
    def reset(self):
        global episode_reward
        episode_reward = 0
        super(Environment, self).reset()
 
        return super(Environment, self).step([actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])
 

# Agent architecture using keras rl
 
def neural_network_model(input, actions):
    model = Sequential()
    # Define CNN model
    print(input)
    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=input))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(3, 3), input_shape=input))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    model.add(Dropout(0.3))
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Reshape((1, 256)))
    # Add some memory
    model.add(LSTM(256))
    model.add(Dense(actions, activation='softmax'))
    model.summary()
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
 
    return model
 
 
def training_game():
    env = Environment(map_name="DefeatRoaches", visualize=True, game_steps_per_episode=150,
                      agent_interface_format=features.AgentInterfaceFormat(
                          feature_dimensions=features.Dimensions(screen=64, minimap=32)
                      ))
 
    input_shape = (_SIZE, _SIZE, 1)
    nb_actions = 12  # Number of actions
 
    model = neural_network_model(input_shape, nb_actions)
    memory = SequentialMemory(limit=5000, window_length=_WINDOW_LENGTH)
 
    processor = SC2Proc()
 
    # Policy
 
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=1, value_min=0.2, value_test=.0,
                                  nb_steps=1e2)
 
    # Agent
 
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, enable_double_dqn=True,
                   enable_dueling_network=True,
                   nb_steps_warmup=500, target_model_update=1e-2, policy=policy,
                   batch_size=150,
                   processor=processor,
                   delta_clip=1)
 
    dqn.compile(Adam(lr=.001), metrics=["mae", "acc"])

    # Tensorboard callback

    callbacks = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                write_graph=True, write_images=False)



    # Save the parameters and upload them when needed

    name = "agent"
    w_file = "dqn_{}_weights.h5f".format(name)
    check_w_file = "train_w" + name + "_weights.h5f"

    if SAVE_MODEL:
        check_w_file = "train_w" + name + "_weights_{step}.h5f"

    log_file = "training_w_{}_log.json".format(name)


    if LOAD_MODEL:
        dqn.load_weights(w_file)

    class Saver(Callback):
        def on_episode_end(self, episode, logs={}):
            if episode % 200 == 0:
                self.model.save_weights(w_file, overwrite=True)

    s = Saver()
    logs = FileLogger('DQN_Agent_log.csv', interval=1)

    dqn.fit(env, callbacks=[callbacks,s,logs], nb_steps=600, action_repetition=2,
            log_interval=1e4, verbose=2)
 
    dqn.save_weights(w_file, overwrite=True)
    dqn.test(env, action_repetition=2, nb_episodes=30, visualize=False)
 
 
if __name__ == '__main__':
    print('FLAGS:')
    for k in FLAGS._flags():
        print(k, FLAGS[k].value)
    print('-'*20)
    training_game()
