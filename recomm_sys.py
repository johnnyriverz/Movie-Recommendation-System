# read a data set of movie reviews and deliver a new recommendation to watch
# follow along to Siraj Raval

import numpy as np
import scipy
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# lets fetch the data and format it

data = fetch_movielens(min_rating = 4.0)

# now print training and testing data

print(repr(data['train']))
print(repr(data['test']))

# lets create a model

model = LightFM(loss = 'warp')

# now train the model

model.fit(data['train'], epochs = 30, num_threads = 2)

def sample_recommendation(model,data,user_ids):
  # first get the number of users and movies in training data
  n_users, n_items = data['train'].shape

  # generate recommendations for each user we input
  for user_id in user_ids:
    # movies they already like
    known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

    # movies our model predicts
    scores = model.predict(user_id, np.arange(n_items))

    # rank them in order of their scores
    top_items = data['item_labels'][np.argsort(-scores)]

    # print out results
    print("User %s" % user_id)
    print("  Known positives: ")

    for x in known_positives[:3]:
      print("              %s" % x)

    print("      Recommended: ")

    for x in top_items[:3]:
      print("              %s" % x)

sample_recommendation(model, data, [3, 25,4])
