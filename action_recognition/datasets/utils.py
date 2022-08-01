
import math
import numpy as np

# change seed to pick up  a different subset of  random samples
seed = 99999
#seed = 12345
#seed = 19
def get_subset_data(video_names,annotations,num_of_examples):

    examples_per_class =   int(math.ceil(num_of_examples / len(set(annotations))))  
    print("original data length", len(video_names) )
    print("subset data", examples_per_class,num_of_examples)

    random_state = np.random.RandomState(seed)

    annotations = np.array(annotations)
    video_names = np.array(video_names )

    subset_video_names = []
    subset_annotations = []
    for class_label in set(annotations): # sample uniformly from each class

          subset_indexes = np.where(annotations == class_label)
          #print("shape",subset_indexes[0].shape[0])

          if examples_per_class > subset_indexes[0].shape[0]:

                           ran_indicies = np.array(random_state.choice(subset_indexes[0].shape[0],subset_indexes[0].shape[0],replace=False))
          else:
                           ran_indicies = np.array(random_state.choice(subset_indexes[0].shape[0],examples_per_class,replace=False))

          indicies_100 = (subset_indexes[0][ran_indicies])
          temp_annotations =annotations[indicies_100]
          temp_names=  video_names[indicies_100]
          subset_video_names.extend(temp_names)
          subset_annotations.extend(temp_annotations)
    video_names  = list(subset_video_names)
    annotations = list(subset_annotations)
    print(len(video_names),len(annotations))
    #print(video_names)
    #print(annotations)
    return video_names, annotations
