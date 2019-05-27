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

    def _convert_int(self, x):
        try:
            return int(x)
        except:
            return np.nan

    def __init__(self):
        self.similarity = None
        self.similarity_type = None
        self.list_size = None
        self.user_training = None
        self.user_training_type = None

        print("Loading metadata...")
        self.metadata = pd.read_csv(RecommenderSystem.METADATA_PATH, low_memory=False)
        # Replace NaN with an empty string
        self.metadata['overview'] = self.metadata['overview'].fillna('')

        # Load keywords and credits
        print("Loading credits and keywords...")
        credits = pd.read_csv(RecommenderSystem.CREDITS_PATH)
        keywords = pd.read_csv(RecommenderSystem.KEYWORDS_PATH)

        # Remove rows with bad IDs.
        print("Preprocessing metadata...")
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

        # Dataframe con los ratings
        print("Loading ratings...")
        self.ratings = pd.read_csv(RecommenderSystem.RATINGS_PATH)

        # Índices para mapear entre ratings y metadata
        self.id_map = pd.read_csv(RecommenderSystem.LINKS_PATH)[['movieId', 'tmdbId']]
        self.id_map['tmdbId'] = self.id_map['tmdbId'].apply(self._convert_int)
        self.id_map.columns = ['movieId', 'id']
        # mezclamos por id con los datos
        self.id_map = self.id_map.merge(self.metadata[['title', 'id', 'clean_title']], on='id').set_index('title')
        self.indices_map = self.id_map.set_index('id')
        self.indices_map = self.indices_map.loc[~self.indices_map.index.duplicated()]

        reader = Reader()

        self.rating_ds = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], reader)

    def get_metadata(self):
        return self.metadata

    # Métrica de simililtud basada en las descripciones de las películas.
    def set_overview_similarity_metric(self):
        if self.similarity_type != 'overview':
            self.similarity_type = 'overview'
            # Ocupan demasiada memoria, no es posible mantener más de una a la vez.
            del self.similarity
            gc.collect()  # Recolector de basura.
            print("Creating new similarity matrix...")
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

            print("Creating new similarity matrix...")
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

    def get_content_recommendations_by_index(self, index, top=10):
        if self.similarity is None:
            raise ValueError("A similarity metric must be defined.")
        if isinstance(index, int):
            index = [index]

        nmovies = len(index)

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(np.asarray(self.similarity[index].todense().max(axis=0)).ravel()))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[nmovies:(top + nmovies)]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        return movie_indices

    def get_content_recommendations(self, title, top=10):
        if self.similarity is None:
            raise ValueError("A similarity metric must be defined.")
        if isinstance(title, str):
            title = [title]

        clean_title = [self._clean_title(t) for t in title]
        # Get the index of the movie that matches the title
        idx = self.indices[clean_title]

        movie_indices = self.get_content_recommendations_by_index(idx, top)

        # Return the top 10 most similar movies
        return self.metadata['title'].iloc[movie_indices]

    def set_svd_user_training(self):
        if self.user_training_type != 'svd':
            self.user_training_type = 'svd'
            print("Training SVD...")
            self.user_training = SVD()
            train = self.rating_ds.build_full_trainset()
            self.user_training.fit(train)

    def get_hybrid_recommendations_by_index(self, userId, index, top=10, content_top=25):
        if self.similarity is None:
            raise ValueError("A similarity metric must be defined.")
        if isinstance(index, int):
            index = [index]

        nmovies = len(index)
        # para el hibrido necesitamos saber el indice de la pelicula en metadata original
        # tras ello, buscamos las peliculas con mayor similitud según el coseno
        # devolvemos las peliculas con mejor estimacion de puntuacion usando svd

        # tmdbId = id_map.loc[clean_title]['id']
        # movie_id = id_map.loc[clean_title]['movieId']

        sim_scores = list(enumerate(np.asarray(self.similarity[index].todense().max(axis=0)).ravel()))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[nmovies:(content_top + nmovies)]

        movie_indices = [i[0] for i in sim_scores]
        movies = self.metadata.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]

        movies['estimation'] = movies['id'].apply(lambda x: self.user_training.predict(userId, self.indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('estimation', ascending=False)
        return list(movies.index)[:top], movies['estimation'].head(top)

    def get_hybrid_recommendations(self, userId, title, top=10, content_top=25):
        if self.similarity is None:
            raise ValueError("A similarity metric must be defined.")
        if isinstance(title, str):
            title = [title]

        clean_title = [self._clean_title(t) for t in title]
        # Get the index of the movie that matches the title
        idx = self.indices[clean_title]

        movie_indices, estimations = self.get_hybrid_recommendations_by_index(userId, idx, top, content_top)
        results = pd.concat([self.metadata['title'].iloc[movie_indices], estimations], axis=1)

        return results
        
    def get_collaborative_recommendations_by_index(self, userId, top=10):
        # nos quedamos unicamente con las películas que no haya visto userId.
        # tomamos unicamente aquellas peliculas que no ha visto
        condition = self.ratings[self.ratings.iloc[:,0] != 1]
        movie_indices = condition['movieId'].drop_duplicates()
        
        # buscamos en el dataset con todos los datos las peliculas cuyo Id corresponde con el de la condición anterior y obtenemos los datos deseados
        movies = self.metadata.loc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]
        movies = movies.dropna()
        movies['estimation'] = movies['id'].apply(lambda x: self.user_training.predict(userId, self.indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('estimation', ascending=False)
        return list(movies.index)[:top], movies['estimation'].head(top)
        

    def get_collaborative_recommendations(self,userId, top=10):
        movie_indices, estimations = self.get_collaborative_recommendations_by_index(userId,top)
        results = pd.concat([self.metadata['title'].iloc[movie_indices], estimations], axis=1)

        return results



recsys = RecommenderSystem() # Inicializar
recsys.set_svd_user_training()
print(recsys.get_collaborative_recommendations(1))
