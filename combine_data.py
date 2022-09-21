import numpy as np
import scipy.io
import pickle

LFP = np.load('LFP.npy', allow_pickle=True)
print(LFP.shape)
LFP_params = np.load('LFP_params.npy', allow_pickle=True)
LFP_params = LFP_params[()]
displacement_map = LFP_params['displacement_map']
del LFP_params['displacement_map']
data = scipy.io.loadmat('task_data')
print(data['spikes_share'].shape)
print(data['spikes_identity'].shape)

save_data = {'LFP': LFP, 'LFP_params': LFP_params, 'displacement_map': displacement_map, 'spikes': data['spikes_share'],
             'spikes_identity': data['spikes_identity'],
             'task': {'target_1_on': data['target_1_on'], 'target_1_off': data['target_1_off'],
                      'target_2_on': data['target_2_on'], 'target_2_off': data['target_2_off'],
                      'perturbation_1_on': data['perturbation_1_on'],
                      'perturbation_1_off': data['perturbation_1_off'],
                      'perturbation_2_on': data['perturbation_2_on'],
                      'perturbation_2_off': data['perturbation_2_off'],
                      'reward': data['reward']}}

with open('012622_data.pickle', 'wb') as handle:
    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(LFP_params)