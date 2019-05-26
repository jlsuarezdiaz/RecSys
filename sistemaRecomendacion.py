import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from surprise import Reader, Dataset, SVD, evaluate

import gc


class RecommenderSystem:

    MOVIES_FOLDER = 'the-movies-dataset/'
    
    METADATA_PATH = MOVIES_FOLDER + 'movies_metadata.csv'

    CREDITS_PATH = MOVIES_FOLDER + 'credits.csv'

    KEYWORDS_PATH = MOVIES_FOLDER + 'keywords.csv'

    RATINGS_PATH = MOVIES_FOLDER + 'ratings_small.csv'
    
    LINKS_PATH = MOVIES_FOLDER + 'links.csv'
    # Get the director's name from the crew feature. If director is not listed, return NaN
    def _get_director(self, x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    # Returns the list top 3 elements or entire list; whichever is more.
    def _get_list(self, x, n=3):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > n:
                names = names[:n]
            return names

        # Return empty list in case of missing/malformed data
        return []

    # Function to convert all strings to lower case and strip names of spaces
    def _clean_data(self, x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "").replace(".", "")) for i in x]
        else:
            # Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", "").replace(".", ""))
            else:
                return ''

    def _clean_title(self, title):
        return str.lower(title.replace(" ", "").replace(".", "").replace(",", "").replace(":", "").replace(";", ""))

    def _create_soup(self, x):
        return ' '.join(x['top_keywords']) + ' ' + ' '.join(x['top_cast']) + ' ' + x['clean_director'] + ' ' + ' '.join(x['top_genres'])

    def __init__(self):
        self.similarity = None
        self.similarity_type = None
        self.list_size = None
        self.metadata = pd.read_csv(RecommenderSystem.METADATA_PATH, low_memory=False)
        # Replace NaN with an empty string
        self.metadata['overview'] = self.metadata['overview'].fillna('')

        # Load keywords and credits
        credits = pd.read_csv(RecommenderSystem.CREDITS_PATH)
        keywords = pd.read_csv(RecommenderSystem.KEYWORDS_PATH)
        # Remove rows with bad IDs.
        self.metadata = self.metadata.drop([19729, 19730, 29502, 29503, 35586, 35587])

        # Convert IDs to int. Required for merging
        keywords['id'] = keywords['id'].astype('int')
        credits['id'] = credits['id'].astype('int')
        self.metadata['id'] = self.metadata['id'].astype('int')

        # Merge keywords and credits into your main metadata dataframe
        self.metadata = self.metadata.merge(credits, on='id')
        self.metadata = self.metadata.merge(keywords, on='id')

        # Parse the stringified features into their corresponding python objects
        features = ['cast', 'crew', 'keywords', 'genres']
        for feature in features:
            self.metadata[feature] = self.metadata[feature].apply(literal_eval)

        # Define new director, cast, genres and keywords features that are in a suitable form.
        self.metadata['director'] = self.metadata['crew'].apply(self._get_director)

        self.features = ['cast', 'keywords', 'genres']
        for feature in features:
            self.metadata[feature] = self.metadata[feature].apply(self._get_list, n=100)

        # Apply clean_data function to your features.
        features = ['cast', 'keywords', 'director', 'genres']

        for feature in features:
            self.metadata['clean_' + feature] = self.metadata[feature].apply(self._clean_data)

        # Índice para acceder a las películas por nombre.
        self.metadata['clean_title'] = self.metadata['title'].apply(self._clean_title)
        self.metadata = self.metadata.reset_index()
        self.indices = pd.Series(self.metadata.index, index=self.metadata['clean_title'])

    def get_metadata(self):
        return self.metadata

    # Métrica de simililtud basada en las descripciones de las películas.
    def set_overview_similarity_metric(self):
        if self.similarity_type != 'overview':
            self.similarity_type = 'overview'
            # Ocupan demasiada memoria, no es posible mantener más de una a la vez.
            del self.similarity
            gc.collect()  # Recolector de basura.

            # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
            tfidf = TfidfVectorizer(stop_words='english')
            # Construct the required TF-IDF matrix by fitting and transforming the data
            tfidf_matrix = tfidf.fit_transform(self.metadata['overview'])
            # Compute the cosine similarity matrix
            self.similarity = linear_kernel(tfidf_matrix, dense_output=False)

    # Métrica de similitud basada en créditos, géneros y keywords (cgk)
    def set_cgk_similarity_metric(self, list_size=3):
        if self.similarity_type != 'cgk' or self.list_size != list_size:
            self.similarity_type = 'cgk'
            self.list_size = list_size
            # Ocupan demasiada memoria, no es posible mantener más de una a la vez.
            del self.similarity
            gc.collect()  # Recolector de basura.

            features = ['cast', 'keywords', 'genres']

            for feature in features:
                self.metadata['top_' + feature] = self.metadata['clean_' + feature].apply(lambda x: x[:self.list_size])
            self.metadata['soup'] = self.metadata.apply(self._create_soup, axis=1)

            # Create the count matrix
            count = CountVectorizer(stop_words='english')
            count_matrix = count.fit_transform(self.metadata['soup'])

            # Compute the Cosine Similarity matrix based on the count_matrix
            self.similarity = cosine_similarity(count_matrix, dense_output=False)

    def clear_similarity_metric(self):
        self.similarity_type = None
        self.list_size = None
        del self.similarity
        gc.collect()

    def get_content_recommendations(self, title, top=10):
        if self.similarity is None:
            raise ValueError("A similarity metric must be defined.")

        clean_title = self._clean_title(title)
        # Get the index of the movie that matches the title
        idx = self.indices[clean_title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(np.asarray(self.similarity[idx].todense().max(axis=0)).ravel()))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:(top + 1)]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return self.metadata['title'].iloc[movie_indices]

    
    # por acabar
    def get_collaborative_recommendations(self, userId, movieId, movieRating, nfolds = 5):
        reader = Reader()

        # Load ratings
        ratings = pd.read_csv(RecommenderSystem.RATINGS_PATH)
        ratings.head()
      
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        data.split(n_folds=nfolds)
      
        # perform SVD decomposition
        svd = SVD()
      
        # evaluation of SVD decomposition
        evaluate(svd, data, measures=['RMSE', 'MAE'])

        # training phase
        train = data.build_full_trainset()
        svd.train(train)
      
        #test phase
        ratings[ratings['userId'] == userId]
      
        return svd.predict(userId,movieId,movieRating)
    
    
    def convert_int(self,x):
        try:
            return int(x)
        except:
            return np.nan


    def get_hybrid_recommendations(self, userId, title):
        if self.similarity is None:
            raise ValueError("A similarity metric must be defined.")
        
        # tomamos los índics de las películas 
        
        # esto puede servir simplemente para el uso de SVD
        id_map = pd.read_csv(RecommenderSystem.LINKS_PATH)[['movieId', 'tmdbId']]
        id_map['tmdbId'] = id_map['tmdbId'].apply(self.convert_int)
        id_map.columns = ['movieId', 'id']
        # mezclamos por id con los datos
        id_map = id_map.merge(self.metadata[['title', 'id','clean_title']], on='id').set_index('title')
        indices_map = id_map.set_index('id')
        indices_map = indices_map.loc[~indices_map.index.duplicated()]

        reader = Reader()

        # Load ratings
        ratings = pd.read_csv(RecommenderSystem.RATINGS_PATH)
        # ratings.head()
      
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        # data.split(n_folds=nfolds)

        # perform SVD decomposition
        svd = SVD()

        # evaluation of SVD decomposition
        # evaluate(svd, data, measures=['RMSE', 'MAE'])

        # training phase
        train = data.build_full_trainset()
        svd.fit(train)

        # print(id_map.head(5))
        # print(indices_map.head(5))
        
        # esto si sirve
        clean_title = self._clean_title(title)
        # Get the index of the movie that matches the title
        idx = self.indices[clean_title]
        
        # para el hibrido necesitamos saber el indice de la pelicula en metadata original
        # tras ello, buscamos las peliculas con mayor similitud según el coseno
        # devolvemos las peliculas con mejor estimacion de puntuacion usando svd

        # esto dos lineas no las usa para nada despues en el tutorial
        #tmdbId = id_map.loc[clean_title]['id']
        #movie_id = id_map.loc[clean_title]['movieId']
        
        sim_scores = list(enumerate(np.asarray(self.similarity[idx].todense().max(axis=0)).ravel()))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]

        movie_indices = [i[0] for i in sim_scores]
        movies = self.metadata.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]
        print(movies)
        print(indices_map)
        movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('est', ascending=False)

        return movies.head(10)


recsys = RecommenderSystem()
recsys.set_overview_similarity_metric()  # Elegir métrica de similaridad
recsys.get_hybrid_recommendations(1,'Toy Story')
print(recsys.get_hybrid_recommendations(1,'Toy Story')) # da vacío :S