import gym
import numpy as np
import matplotlib.pyplot as plt


class CartPoleEnv:
    def __init__(self, agent, game_epoch, is_render_image, is_verbose, is_train):
        # construct gym cart pole environment
        self._env = gym.make('CartPole-v0')
        self._env = self._env.unwrapped

        # define the reinforcement learning agent
        self._agent = agent

        # number of games to play
        self._game_epoch = game_epoch

        # display the game on the screen
        self._is_render_image = is_render_image

        # display the reward information of each game
        self._is_verbose = is_verbose

        # train the model or just test the model
        self._is_train = is_train

    def run(self):
        # record each game's reward and step
        reward_list = []
        step_list = []

        # play No.e game
        for e in range(self._game_epoch):

            # print information if we have enough games
            if len(reward_list) % 1000 == 0 and len(reward_list) > 0:
                print('> average recent game rewards: ' + str(np.average(reward_list[-1000:])))

            if len(step_list) % 1000 == 0 and len(step_list) > 0:
                print('> average recent game steps: ' + str(np.average(step_list[-1000:])))

            # receive initial observation
            observation_current = self._env.reset()

            # performances of this game epoch
            reward_this_epoch = 0
            step_this_epoch = 0

            # run this game till end
            while True:
                # render the game to screen
                if self._is_render_image:
                    self._env.render()

                # reinforcement learning agent choose which action to take
                action = self._agent.choose_action(observation_current)

                # environment receives the action the agent took
                observation_next, reward, is_done, info = self._env.step(action)

                # adjust the reward for better learning
                x, _, theta, _ = observation_next
                r1 = (self._env.x_threshold - abs(x)) / self._env.x_threshold - 0.8
                r2 = (self._env.theta_threshold_radians - abs(theta)) / self._env.theta_threshold_radians - 0.5
                reward = r1 + r2

                if self._is_train:
                    # store the data for training
                    self._agent.store_train_data(observation_current, action, reward, observation_next, is_done)

                    # train the agent when data is enough
                    if self._agent.have_enough_new_data():
                        self._agent.train()

                    # clear the train data if it is too many
                    self._agent.clear_excessive_data()
                else:
                    pass

                # update game information
                reward_this_epoch += reward
                step_this_epoch += 1

                # if the game is finished, we reset the game to restart.
                # if the game is not finished, we keep on playing in the current game.
                if is_done:
                    if self._is_verbose:
                        print('> reward for the {0}th game is {1}.'.format(e+1, reward_this_epoch))
                        print('> step for the {0}th game is {1}.'.format(e+1, step_this_epoch))
                    break
                else:
                    observation_current = observation_next
                    continue

            # record the performance of this game epoch
            reward_list.append(reward_this_epoch)
            step_list.append(step_this_epoch)

        # plot the reward list
        plt.plot(reward_list, label='reward', color='lightgray')
        plt.legend(loc=1)
        plt.xlabel('Game Number')
        plt.ylabel('Reward')
        plt.show()

        # plot the step list
        plt.plot(step_list, label='step', color='lightgray')
        plt.legend(loc=1)
        plt.xlabel('Game Number')
        plt.ylabel('Step')
        plt.show()
