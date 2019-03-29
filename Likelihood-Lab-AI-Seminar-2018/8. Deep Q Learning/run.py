from dqn import DeepQNet
from env import CartPoleEnv


if __name__ == '__main__':
    dqn = DeepQNet(n_actions=2,
                   n_features=4,
                   learning_rate=2e-3,
                   momentum=1e-1,
                   l2_penalty=1e-4,
                   fit_epoch=10,
                   batch_size=10,
                   discount_factor=0.9,
                   e_greedy=0.1,
                   memory_size=10000)

    env = CartPoleEnv(agent=dqn,
                      game_epoch=50000,
                      is_render_image=False,
                      is_verbose=False,
                      is_train=True)

    env.run()
