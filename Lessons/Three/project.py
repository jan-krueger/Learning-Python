import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch data and format it
data = fetch_movielens(min_rating=4.0)

#printtraining and testing data
print(repr(data['train']))
print(repr(data['test']))

# create model
# Loss means the loss-function, and it measures the difference between the models
# prediction and the desired output, we wanna minimize it during training so the
# models gets more accurate over time
# warp:
# - weighted
# - approximate
# - rank
# - pairwise
model = LightFM(loss='warp')

#train the model
model.fit(data['train'], epochs=30, num_threads = 6)

def sample_recommendation(model, data, user_ids):

    #number of users and movies in the training data
    n_users, n_items = data['train'].shape

    #generate recommendations for each user we input
    for user_id in user_ids:

        #movies they alread like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))

        #rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        #print
        print('User %s' % user_id)
        print('     Known positives:')
        for e in known_positives[:3]:
            print('         %s' % e)

        print('     Recommended:')
        for e in top_items[:3]:
            print('         %s' % e)

# Run the app
sample_recommendation(model, data, [1, 500, 600])
