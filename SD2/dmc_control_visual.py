import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'
normalization = 'normalization'
from wrappers import DeepMindControl



def plot(rela):
    plt.matshow(rela, cmap='RdBu_r', vmax=1000,vmin=-1000)
    plt.colorbar(extend='both')
    plt.show()


if __name__ == '__main__':
    for env_item in ['cheetah_run', 'walker_run', 'humanoid_walk', 'hopper_hop', 'finger_spin', 'reacher_easy']:
        if not os.path.exists("./rela_10000_visual/{}/".format(env_item)):
            os.makedirs("./rela_10000_visual/{}/".format(env_item))
        env = DeepMindControl(env_item)
        action_size = env.action_space.shape[0]
        state_size = env.reset()['image'].reshape(-1).shape[0]
        print("=================={}==================".format(env_item))
        print("state_size:", state_size, "action_size:", action_size)

        action_list = []
        delta_s_list = []
        step = 0

        while True:
            last_state = env.reset()['image'].reshape(-1)
            episode_step = 0
            while episode_step < 500:
                episode_step += 1
                action = (np.random.rand(action_size) - 0.5) * 2
                action_list.append(action)
                state = env.step(action)[0]['image'].reshape(-1)
                delta_s = state - last_state
                last_state = state
                delta_s_list.append(delta_s)
                step += 1
            if step > 100000:
                break
        print("step:", step)
        data_size = len(action_list)

        action_array = np.array(action_list).reshape(data_size, action_size)
        delta_s_array = np.array(delta_s_list).reshape(data_size, state_size)

        if normalization == 'normalization':
            print('normalize')
            action_array_mean = np.mean(action_array, 0)
            delta_s_array_mean = np.mean(delta_s_array, 0)
            action_array_std = np.std(action_array, 0)
            delta_s_array_std = np.std(delta_s_array, 0)
            action_array = (action_array - action_array_mean) / action_array_std
            delta_s_array = (delta_s_array - delta_s_array_mean) / delta_s_array_std
        action_array = action_array.T
        delta_s_array = delta_s_array.T
        sum_rela = np.zeros((action_size, state_size))


        for action_id in range(action_size):
            for state_id in range(state_size):
                data = np.vstack([action_array[action_id], delta_s_array[state_id]])
                rela = np.corrcoef(data)
                if rela[0][1] - rela[1][0] > 0.0000000001:
                    print(rela)
                sum_rela[action_id][state_id] = rela[0][1]

        print(env_item, sum_rela)
        plt.matshow(sum_rela, cmap='RdBu_r', vmax=1, vmin=-1)
        plt.colorbar(extend='both')
        plt.savefig('./rela_10000_visual/{}/{}.png'.format(env_item, normalization))

        data_df = pd.DataFrame(sum_rela)
        writer = pd.ExcelWriter('./rela_10000_visual/{}/{}.xlsx'.format(env_item, normalization))
        data_df.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()
        # plot(sum_rela * 1000)
