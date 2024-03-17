# эти библиотеки нам уже знакомы
import pandas as pd
import numpy as np

# модуль sparse библиотеки scipy понадобится
# для работы с разреженными матрицами (об этом ниже)
from scipy.sparse import csr_matrix

# из sklearn мы импортируем алгоритм k-ближайших соседей
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
print(ratings.info())
print(movies.info())
print(ratings.head())
print(movies.head())
movies.drop(['genres'], inplace=True, axis=1)
print(movies.head(3))
ratings.drop(['timestamp'], inplace=True, axis=1)
print(ratings.head(3))
user_item_matrix = ratings.pivot(index='movieId', columns='userId', values='rating')
print(user_item_matrix.head(3))
user_item_matrix.fillna(0, inplace=True)
print(user_item_matrix)
print(user_item_matrix.shape)

users_votes = ratings.groupby('userId')['rating'].agg('count')
movies_votes = ratings.groupby('movieId')['rating'].agg('count')

user_mask = users_votes[users_votes > 50].index
movie_mask = movies_votes[movies_votes > 10].index
print("!!!!")
user_item_matrix = user_item_matrix.loc[movie_mask, :]
print(user_item_matrix.shape)
user_item_matrix = user_item_matrix.loc[:, user_mask]
print(user_item_matrix.shape)

csr_data = csr_matrix(user_item_matrix.values)
print(csr_data[:7, :10])

user_item_matrix = user_item_matrix.rename_axis(None, axis=1).reset_index()
print(user_item_matrix.head(3))

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

recommendations = 10
search_word = 'Matrix'

movie_search = movies[movies['title'].str.contains(search_word)]
print(movie_search)

movie_id = movie_search.iloc[0]['movieId']
movie_id = user_item_matrix[user_item_matrix['movieId'] == movie_id].index[0]

print(movie_id)

distances, indices = knn.kneighbors(csr_data[movie_id], n_neighbors=recommendations + 1)

indices_list = indices.squeeze().tolist()
distances_list = distances.squeeze().tolist()

indices_distances = list(zip(indices_list, distances_list))
print(type(indices_distances[0]))
print(indices_distances[:3])

indices_distances_sorted = sorted(indices_distances, key=lambda x: x[1], reverse=False)
indices_distances_sorted = indices_distances_sorted[1:]
print(indices_distances_sorted)

recom_list = []


for ind_dist in indices_distances_sorted:

    matrix_movie_id = user_item_matrix.iloc[ind_dist[0]]['movieId']


    id = movies[movies['movieId'] == matrix_movie_id].index


    title = movies.iloc[id]['title'].values[0]
    dist = ind_dist[1]


    recom_list.append({'Title': title, 'Distance': dist})

print(recom_list[0])

recom_df = pd.DataFrame(recom_list, index=range(1, recommendations + 1))
print(recom_df)