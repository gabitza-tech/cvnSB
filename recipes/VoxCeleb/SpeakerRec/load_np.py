import numpy as np
import sys

file = sys.argv[1]

enroll_dict = np.load(file, allow_pickle=True)

feat = enroll_dict['concat_features']

labels = enroll_dict['concat_labels']

slices = enroll_dict['concat_slices']

patchs = enroll_dict['concat_patchs']

print(feat.shape)
#for i in range(200,208):
   #print([feat[i],labels[i],slices[i],patchs[i]])

