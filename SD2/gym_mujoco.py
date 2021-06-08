# from dm_control import suite
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import gym

tasks = [('HalfCheetah', 'v2'), ('Ant', 'v2'), ('Hopper', 'v2'), ('Walker2d', 'v2')]

if __name__ == '__main__':
    for domain_name, task_name in tasks:
        if not os.path.exists("./rela_gym/{}_{}/".format(domain_name, task_name)):
            os.makedirs("./rela_gym/{}_{}/".format(domain_name, task_name))
        env = gym.make(domain_name + '-' + task_name)
        action_size = env.action_space.shape[0]
        state_size = env.observation_space.shape[0]
        print("=================={}_{}==================".format(domain_name, task_name))
        print("state_size:", state_size, "action_size:", action_size)

        action_list = []
        delta_s_list = []
        step = 0
        episode_num = 0
        while True:
            last_state = env.reset()
            episode_step = 0
            while episode_step < 1000:
                # print(step)
                episode_step += 1
                action = (np.random.rand(action_size) - 0.5) * 2
                action_list.append(action)
                return_tuple = env.step(action)
                state = return_tuple[0]
                delta_s = state - last_state
                last_state = state
                delta_s_list.append(delta_s)
                step += 1
                # if return_tuple[1] == True:
                #     break
            episode_num += 1
            if step > 10000:
                break
        print(step, episode_num)
        print("step:", step)
        data_size = len(action_list)

        action_array = np.array(action_list).reshape(data_size, action_size)
        delta_s_array = np.array(delta_s_list).reshape(data_size, state_size)
        action_array = action_array.T
        delta_s_array = delta_s_array.T
        sum_rela = np.zeros((action_size, state_size))

        for action_id in range(action_size):
            for state_id in range(state_size):
                data = np.vstack([action_array[action_id], delta_s_array[state_id]])
                rela = np.corrcoef(data)
                sum_rela[action_id][state_id] = rela[0][1]

        print(domain_name, task_name, sum_rela)
        plt.matshow(sum_rela, cmap='RdBu_r', vmax=1, vmin=-1)
        plt.colorbar(extend='both')
        plt.savefig('./rela_gym/{}_{}/{}.png'.format(domain_name, task_name, 0))

        data_df = pd.DataFrame(sum_rela)
        writer = pd.ExcelWriter('./rela_gym/{}_{}/{}.xlsx'.format(domain_name, task_name, 0))
        data_df.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()

