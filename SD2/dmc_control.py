from dm_control import suite
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd


def get_state(time_step, keys):
    return_list = []
    for key in keys:
        if type(time_step.observation[key]) is np.ndarray:
            return_list.append(time_step.observation[key].flatten())
        else:
            return_list.append(np.array(time_step.observation[key]).reshape(1))
    return_array = np.concatenate(return_list)
    return return_array


if __name__ == '__main__':
    for domain_name, task_name in suite.BENCHMARKING:
        if not os.path.exists("./rela_dmc/{}_{}/".format(domain_name, task_name)):
            os.makedirs("./rela_dmc/{}_{}/".format(domain_name, task_name))
        env = suite.load(domain_name, task_name)
        action_size = env.action_spec().shape[0]
        information_get_step = env.reset()
        state_keys = information_get_step.observation.keys()
        state_size = get_state(information_get_step, state_keys)
        state_size = state_size.shape[0]
        print("=================={}_{}==================".format(domain_name, task_name))
        print("state_size:", state_size, "action_size:", action_size, "state_keys:", state_keys)
        action_list = []
        delta_s_list = []
        step = 0
        while True:
            TimeStep = env.reset()
            last_state = get_state(TimeStep, state_keys)
            episode_step = 0
            while episode_step < 1000:
                episode_step += 1
                action = (np.random.rand(action_size) - 0.5) * 2
                action_list.append(action)
                TimeStep = env.step(action)
                state = get_state(TimeStep, state_keys)
                delta_s = state - last_state
                last_state = state
                delta_s_list.append(delta_s)
                step += 1
                if TimeStep.last():
                    break
            if step > 20000:
                break
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
        plt.savefig('./rela_dmc/{}_{}/{}.png'.format(domain_name, task_name, 0))

        data_df = pd.DataFrame(sum_rela)
        writer = pd.ExcelWriter('./rela_dmc/{}_{}/{}.xlsx'.format(domain_name, task_name, 0))
        data_df.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()
