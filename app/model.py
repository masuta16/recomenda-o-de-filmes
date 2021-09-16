from operator import itemgetter
import io
import json

from surprise import KNNWithMeans, KNNWithZScore, KNNBasic, KNNBaseline
from surprise import Dataset, accuracy
from surprise.model_selection import cross_validate, train_test_split
from surprise import get_dataset_dir


# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Spliting data in train (85%) and test (15%)
trainset, testset = train_test_split(data, test_size=.15)

# Using KNN with k = 40
algo = KNNBaseline(k=25, sim_options={'name': 'pearson_baseline', 'user_based': True})

# Run 10-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)

# Training the model
algo.fit(trainset)

# Get the movies that user doesn't watch.
def get_unwatched_movies(userId):
    not_watched = []
    for item in testset:
        if item[1][0] != userId:
            not_watched.append(item)
    return not_watched

# convert raw ids into movie names
def read_item_names():
    file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.item'
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]

    return rid_to_name

# convert raw ids into user data
def id_to_user():
    file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.user'
    rid_to_user = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_user[line[0]] = {}
            rid_to_user[line[0]]['Identificador'] = line[0]
            rid_to_user[line[0]]['Idade'] = line[1]
            rid_to_user[line[0]]['Genero'] = line[2]
    return rid_to_user

# Get 10 recommended movies
def top_10_recommended_movies(userId):
    rid_to_name = read_item_names()

    not_watched = get_unwatched_movies(userId)

    predicted_rating = []

    for item in not_watched:
        movie = []
        movie.append(item[1])
        name = rid_to_name[str(item[1])]
        movie.append(name)
        predition = algo.predict(userId, item[1])
        movie.append(float(predition.est))

        if movie not in predicted_rating:
            predicted_rating.append(movie)
    return sorted(predicted_rating, key=itemgetter(2), reverse = True)[:10]

def get_K_neighbors(userId, n_k = 3):
    n = []
    id_user = id_to_user()
    neighbors =  algo.get_neighbors(userId, k = n_k)
    print (neighbors)
    for item in neighbors:
        n.append(id_user[str(item)])
    return n

def list_to_json(movies):
    l = []
    for item in movies:
        d = {}
        d['filme'] = item[1]
        l.append(d)
    return l

def recommended_json(userId):
    d = top_10_recommended_movies(str(userId))
    #d['movies'] = list_to_json(top_10_recommended_movies(str(userId)))
    #d['users'] = get_K_neighbors(userId, 3)
    return d

def main():
    recommended_json(userId)

if __name__ == "__main__":
    main()
