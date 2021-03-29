import pickle
import matplotlib.pyplot as plt
import numpy as np


# trainings_model = 0
#
with open("B_M_I_Y_C_agent.pt", "rb") as file:
    trainings_model = pickle.load(file)
trainings_model.eval()
trainings_model.cpu()

with open("B_M_I_Y_C_agent_cpu.pt", "wb") as file:
    pickle.dump(trainings_model, file)





# plt.figure(figsize=(17, 5))
#
# plt.subplot(131)
# plt.title('Steps per round')
# plt.xlabel('Episode')
# plt.ylabel('Step')
# plt.ylim([0, 410])
# plt.ion()
# ax = plt.gca()
#
#
# plt.subplot(132)
# plt.title('Rewards game')
# plt.xlabel('Episode')
# plt.ylabel('Rewards')
# plt.ylim([0, 30])
# plt.ion()
# ax = plt.gca()
#
# plt.subplot(133)
# plt.title('Score after 1000 round')
# #plt.xlabel('Episode')
# plt.ylabel('Score')
# plt.ylim([0, 100])
# plt.ion()
# ax = plt.gca()
#
#
# plt.suptitle('Q-Net Test')


# plt.subplot(131)
# plt.plot(np.arange(1, 6), (200,300,200,400,100))
# plt.xticks(np.arange(1, 6, 1))
#
# plt.subplot(132)
# plt.plot(np.arange(1, 6), (20,10,5,5,10))
# plt.xticks(np.arange(1, 6, 1))
#
# plt.subplot(133)
# plt.xlim(0.5, 1.5)
# plt.bar([1], [20], width=0.7)
# plt.xticks([])
# plt.xlabel('1000 rounds')
#
# plt.savefig('AAAAAAAAAAAAAA.png')













# file_name = '19_crate_3peaceful_agents_EVAL'
#
# f = open(file_name+'.txt', "r")
# steps=f.readline()
# rewards=f.readline()
# f.close()
#
# steps = steps.split('\t')
# rewards = rewards.split('\t')
#
#
# steps_list = []
# for step in steps:
#     if step!='\n':
#         steps_list.append(int(step))
#
#
# rewards_list = []
# for reward in rewards:
#     if reward!='':
#         rewards_list.append(int(reward))
#
#
#
#
# #file_name = '19_crate_3peaceful_agents_EVAL'
# file_name = '19_crate_3peaceful_agentsEVAL_New'
#
# f = open(file_name+'.txt', "r")
# steps_f=f.readline()
# rewards_f=f.readline()
# f.close()
#
# steps_f = steps_f.split('\t')
# rewards_f = rewards_f.split('\t')
#
#
# steps_list_f = []
# for step_f in steps_f:
#     if step_f!='\n':
#         steps_list_f.append(int(step_f))
#
#
# rewards_list_f = []
# for reward_f in rewards_f:
#     if reward_f!='':
#         rewards_list_f.append(int(reward_f))
#
#
#
#
# plt.figure(figsize=(20, 5))
#
# plt.subplot(121)
# plt.title('Avg steps per round')
# #plt.xlabel('Episode')
# plt.ylabel('Step')
# plt.ylim([0, 410])
# plt.ion()
# ax = plt.gca()
#
# # plt.subplot(132)
# # plt.title('Rewards game')
# # plt.xlabel('Episode')
# # plt.ylabel('Rewards')
# # plt.ylim([0, 15])
# # plt.ion()
# # ax = plt.gca()
#
# plt.subplot(122)
# plt.title('Sum score after 1000 rounds')
# # plt.xlabel('Episode')
# plt.ylabel('Score')
# plt.ylim([0, 3000])
# plt.ion()
# ax = plt.gca()
#
# plt.suptitle('Q-Net Test vs 3 Rule Based')
#
#
# # plt.subplot(121)
# # #plt.plot(np.arange(1, len(steps_list) + 1), steps_list)
# # plt.bar(np.arange(1, len(steps_list) + 1), steps_list, width=1)
# # plt.xticks(np.arange(0, len(steps_list) + 2, 100))
#
# plt.subplot(121)
# plt.xlim(0.5, 2.5)
# plt.bar([1, 2], [np.average(steps_list), np.average(steps_list_f)], width=0.7, color=['#0055EE', '#EE5500'])
# #plt.xticks([])
# plt.xticks([1, 2], ['CNN OLD', 'CNN NEW'])
# #plt.xticks([1, 2], ['FC NN', 'CNN'])
# plt.xlabel('1000 rounds')
#
#
# # plt.subplot(132)
# # #plt.plot(np.arange(1, len(rewards_list) + 1), rewards_list)
# # plt.bar(np.arange(1, len(rewards_list) + 1), rewards_list, width=1)
# # plt.xticks(np.arange(0, len(rewards_list) + 2, 100))
#
# plt.subplot(122)
# plt.xlim(0.5, 2.5)
# plt.bar([1, 2], [np.sum(rewards_list), np.sum(rewards_list_f)], width=0.7, color=['#0055EE', '#EE5500'])
# #plt.xticks([1, 2], ['FC NN', 'CNN'])
# plt.xticks([1, 2], ['CNN OLD', 'CNN NEW'])
# plt.xlabel('1000 rounds')
#
# plt.savefig('3RuleBased_EVAL_Marinko.png')
#
#
#

