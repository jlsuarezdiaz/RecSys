import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from surprise import Reader, Dataset, SVD, evaluate, KNNBasic
from surprise.dataset import DatasetUserFolds
from metrics import personalization, coverage, intra_list_similarity, novelty, mark, precision, recall
from sklearn.model_selection import KFold
import gc
import random


class RecommenderSystem:
    # Variables estáticas con las direcciones de los ficheros de datos.
    MOVIES_FOLDER = 'the-movies-dataset/'

    METADATA_PATH = MOVIES_FOLDER + 'movies_metadata.csv'

    CREDITS_PATH = MOVIES_FOLDER + 'credits.csv'

    KEYWORDS_PATH = MOVIES_FOLDER + 'keywords.csv'

    RATINGS_PATH = MOVIES_FOLDER + 'ratings_small.csv'

    LINKS_PATH = MOVIES_FOLDER + 'links.csv'

    # Método privado para obtener el director a partir de los datos del equipo de rodaje.
    def _get_director(self, x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    # Función para obtener los top n elementos de una lista.
    def _get_list(self, x, n=3):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > n:
                names = names[:n]
            return names

        # Return empty list in case of missing/malformed data
        return []

    # Función que convierte textos a minúsculas eliminando espacios y signos de puntuación.
    def _clean_data(self, x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "").replace(".", "")) for i in x]
        else:
            if isinstance(x, str):
                return str.lower(x.replace(" ", "").replace(".", ""))
            else:
                return ''

    # Función para limpiar títulos de películas. Elimina signos de puntuación y espacios y convierte a minúsculas.
    def _clean_title(self, title):
        return str.lower(title.replace(" ", "").replace(".", "").replace(",", "").replace(":", "").replace(";", ""))

    # Función para crear una bolsa de palabras a partir de las palabras clave, géneros, director y actores.
    def _create_soup(self, x):
        return ' '.join(x['top_keywords']) + ' ' + ' '.join(x['top_cast']) + ' ' + x['clean_director'] + ' ' + ' '.join(x['top_genres'])

    # Función para convertir a valores enteros. Si la conversión no es posible se devuelve NaN.
    def _convert_int(self, x):
        try:
            return int(x)
        except:
            return np.nan

    # Cálculo del rating ponderado para las recomendaciones por popularidad.
    def _weighted_rating(self, x, m, C):
        v = x['vote_count']
        R = x['vote_average']

        return 0.0 if v < m else (v / (v + m) * R) + (m / (m + v) * C)

    # Constructor.
    def __init__(self):
        self.similarity = None
        self.similarity_type = None
        self.list_size = None
        self.user_training = None
        self.user_training_type = None

        # Lectura del archivo de metadatos.
        print("Loading metadata...")
        self.metadata = pd.read_csv(RecommenderSystem.METADATA_PATH, low_memory=False)
        # Reemplazo de argumentos con valores perdidos por cadenas vacías.
        self.metadata['overview'] = self.metadata['overview'].fillna('')

        # Cargar palabras clave y créditos
        print("Loading credits and keywords...")
        credits = pd.read_csv(RecommenderSystem.CREDITS_PATH)
        keywords = pd.read_csv(RecommenderSystem.KEYWORDS_PATH)

        # Eliminación de películas con información corrupta.
        print("Preprocessing metadata...")
        self.metadata = self.metadata.drop([19729, 19730, 29502, 29503, 35586, 35587])

        # Unimos metadatos, palabras clave y créditos en el mismo dataframe.
        keywords['id'] = keywords['id'].astype('int')
        credits['id'] = credits['id'].astype('int')
        self.metadata['id'] = self.metadata['id'].astype('int')

        # Mezclado de datos
        self.metadata = self.metadata.merge(credits, on='id')
        self.metadata = self.metadata.merge(keywords, on='id')

        # Los datos vienen en cadenas de texto que representan listas o diccionarios. Los convertimos a dichos objetos.
        features = ['cast', 'crew', 'keywords', 'genres']
        for feature in features:
            self.metadata[feature] = self.metadata[feature].apply(literal_eval)

        # Obtenemos director, actores, palabras clave y géneros en un formato tratable
        self.metadata['director'] = self.metadata['crew'].apply(self._get_director)

        self.features = ['cast', 'keywords', 'genres']
        for feature in features:
            self.metadata[feature] = self.metadata[feature].apply(self._get_list, n=100)

        # Limpieza de director, actores, palabras clave y géneros (eliminando espacios y convirtiendo a minúscula) para la posterior tokenización.
        features = ['cast', 'keywords', 'director', 'genres']

        for feature in features:
            self.metadata['clean_' + feature] = self.metadata[feature].apply(self._clean_data)

        # Índice para acceder a las películas por nombre.
        self.metadata['clean_title'] = self.metadata['title'].apply(self._clean_title)
        self.metadata = self.metadata.reset_index()
        self.indices = pd.Series(self.metadata.index, index=self.metadata['clean_title'])

        # Scores de popularidad
        m = self.metadata['vote_count'].quantile(0.9)
        C = self.metadata['vote_average'].mean()
        self.metadata['score'] = self.metadata.apply(self._weighted_rating, axis=1, C=C, m=m)

        # Dataframe con los ratings
        print("Loading ratings...")
        self.ratings = pd.read_csv(RecommenderSystem.RATINGS_PATH)

        print("Indexing...")
        # Índices para mapear entre ratings y metadata
        self.id_map = pd.read_csv(RecommenderSystem.LINKS_PATH)[['movieId', 'tmdbId']]
        self.id_map['tmdbId'] = self.id_map['tmdbId'].apply(self._convert_int)
        self.id_map.columns = ['movieId', 'id']
        # mezclamos por id con los datos
        self.id_map = self.id_map.merge(self.metadata[['title', 'id', 'clean_title']], on='id').set_index('title')
        # índice para id --> movieId
        self.indices_map = self.id_map.set_index('id')
        self.indices_map = self.indices_map.loc[~self.indices_map.index.duplicated()]
        # índice para movieId --> id
        self.di_map = self.id_map.set_index('movieId')
        # índice para id --> metadata.index
        self.meta_map = pd.Series(self.metadata.index, index=self.metadata['id'])

        # Eliminar ratings de películas que no están en la base de datos.
        self.ratings = self.ratings[self.ratings['movieId'].isin(self.di_map.index)]

        reader = Reader()
        # Ratings como objeto Dataset de Surprise.
        self.rating_ds = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], reader)

    # Método que permite acceder a los metadatos de las películas.
    def get_metadata(self):
        return self.metadata

    def get_popularity_recommendations_by_index(self, top=10):
        """
        Obtención de recomendaciones por popularidad (por índices)

        Parameters
        ----------

        top: Número de elementos a devolver.

        Returns
        -------
        Índices de las películas recomendadas y correspondientes scores de popularidad.
        """
        meta_sorted = self.metadata[['vote_count', 'vote_average', 'score']].sort_values('score', ascending=False)
        return list(meta_sorted.index)[:top], meta_sorted['score'].head(top)

    def get_popularity_recommendations(self, top=10):
        """
        Obtención de recomendaciones por popularidad

        Parameters
        ----------

        top: Número de elementos a devolver.

        Returns
        -------
        Dataframe con los títulos de las recomendaciones, total de votos, promedio de votos y scores de popularidad.
        """
        movie_indices, _ = self.get_popularity_recommendations_by_index(top)
        return self.metadata[['title', 'vote_count', 'vote_average', 'score']].loc[movie_indices]

    def get_popularity_recommendations_for_user_by_index(self, userId, top=10, positiveThresh=4.0, ratings=None):
        """
        Obtención de recomendaciones por popularidad (por índices)

        Parameters
        ----------

        top: Número de elementos a devolver.

        positiveThresh: Ignored.

        ratings: Ignored.

        Returns
        -------
        Índices de las películas recomendadas y correspondientes scores de popularidad.
        """
        try:
            return self.get_popularity_recommendations_by_index(top)
        except:
            print("No positive recommendations for this user under this threshold.")
            return pd.DataFrame([], columns=['title', 'similarity'])

    def get_popularity_recommendations_for_user(self, userId, top=10, positiveThresh=4.0, ratings=None):
        """
        Obtención de recomendaciones por popularidad

        Parameters
        ----------

        top: Número de elementos a devolver.

        positiveThresh: Ignored.

        ratings: Ignored.

        Returns
        -------
        Dataframe con los títulos de las recomendaciones, total de votos, promedio de votos y scores de popularidad.
        """
        try:
            return self.get_popularity_recommendations(top)
        except:
            print("No positive recommendations for this user under this threshold.")
            return pd.DataFrame([], columns=['title', 'similarity'])

    # Establece la métrica de simililtud basada en las descripciones de las películas.
    def set_overview_similarity_metric(self):
        if self.similarity_type != 'overview':
            self.similarity_type = 'overview'
            # Ocupan demasiada memoria, no es posible mantener más de una a la vez.
            del self.similarity
            gc.collect()  # Recolector de basura.
            print("Creating new similarity matrix...")
            # Objeto que calcula el score TF-IDF. Elimina palabras de parada del inglés.
            tfidf = TfidfVectorizer(stop_words='english')
            # Construcción de la matriz TF-IDF.
            tfidf_matrix = tfidf.fit_transform(self.metadata['overview'])
            # Cálculo de la similaridad del coseno (al ser sobre TF-IDF es equivalente a un kernel lineal)
            self.similarity = linear_kernel(tfidf_matrix, dense_output=False)

    # Establece la métrica de similitud basada en créditos, géneros y keywords (cgk)
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

            # Matriz de ocurrencias
            count = CountVectorizer(stop_words='english')
            count_matrix = count.fit_transform(self.metadata['soup'])

            # Similaridad del coseno sobre la matriz de ocurrencias.
            self.similarity = cosine_similarity(count_matrix, dense_output=False)

    # Elimima la matriz de similaridad actual, liberando la memoria que ocupa.
    def clear_similarity_metric(self):
        self.similarity_type = None
        self.list_size = None
        del self.similarity
        self.similarity = None
        gc.collect()

    def get_content_recommendations_by_index(self, index, top=10):
        """
        Obtención de las recomendaciones basadas en contenido (por índices).

        Parameters
        ----------

        index: Entero o lista con los índices de películas sobre los que buscar.

        top: Número de elementos a devolver.

        Returns
        -------

        Índices de las películas más similares, junto con los scores de similaridad.
        """
        if self.similarity is None:
            raise ValueError("A similarity metric must be defined.")
        if isinstance(index, int):
            index = [index]

        # nmovies = len(index)

        if index:
            # Semejanzas de todas las películas con las películas indicadas.
            sim_scores = np.array(list(enumerate(np.asarray(self.similarity[index].tocsc().max(axis=0).todense()).ravel())))
            sim_scores[index, 1] = -1.0
            # Ordenación basada en las semejanzas.
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Obtención del top de similaridades.
            sim_scores = sim_scores[:top]
            # Valores de similaridad.
            similarities = [i[1] for i in sim_scores]
            # Índices de películas.
            movie_indices = [i[0] for i in sim_scores]

            return movie_indices, similarities
        else:
            return [], []

    def get_content_recommendations(self, title, top=10):
        """
        Obtención de las recomendaciones basadas en contenido a partir de una lista de películas.

        Parameters
        ----------

        title: String o lista con los títulos de películas sobre los que buscar.

        top: Número de elementos a devolver.

        Returns
        -------

        Dataframe con los títulos de las películas más similares y los scores de similaridad.
        """
        if self.similarity is None:
            raise ValueError("A similarity metric must be defined.")
        if isinstance(title, str):
            title = [title]

        clean_title = [self._clean_title(t) for t in title]
        # Get the index of the movie that matches the title
        idx = list(self.indices[clean_title])
        movie_indices, similarities = self.get_content_recommendations_by_index(idx, top)

        # Return the top 10 most similar movies
        results = pd.DataFrame(list(zip(self.metadata['title'].iloc[movie_indices], similarities)), columns=['title', 'similarity'])
        return results
        # return self.metadata['title'].iloc[movie_indices]

    def get_content_recommendations_for_user_by_index(self, userId, top=10, positiveThresh=4.0, ratings=None):
        """
        Obtención de las recomendaciones basadas en contenido para un usuario concreto (por índices).

        Parameters
        ----------

        userId: id del usuario.

        top: número de elementos a recomendar.

        positiveThresh: umbral de las valoraciones con el que se consideran las películas relevantes para el usuario.

        ratings: Ignored.

        Returns
        -------

        Índices de las películas más similares, junto con los scores de similaridad.
        """
        try:
            return self.get_content_recommendations_by_index(self.get_liked_movies_by_index(userId, positiveThresh, ratings)[0], top)
        except:
            print("No positive recommendations for this user under this threshold.")
            return pd.DataFrame([], columns=['title', 'similarity']), []

    def get_content_recommendations_for_user(self, userId, top=10, positiveThresh=4.0, ratings=None):
        """
        Obtención de las recomendaciones basadas en contenido a partir de una lista de películas.

        Parameters
        ----------

        userId: id del usuario.

        top: número de elementos a recomendar.

        positiveThresh: umbral de las valoraciones con el que se consideran las películas relevantes para el usuario.

        ratings: Ignored.

        Returns
        -------

        Dataframe con los títulos de las películas más similares y los scores de similaridad.
        """
        try:
            return self.get_content_recommendations(self.get_liked_movies(userId, positiveThresh, ratings)['title'], top)
        except:
            print("No positive recommendations for this user under this threshold.")
            return pd.DataFrame([], columns=['title', 'similarity'])

    # Establece SVD como método de entrenamiento y predicción de ratings.
    def set_svd_user_training(self):
        if self.user_training_type != 'svd':
            self.user_training_type = 'svd'
            print("Training SVD...")
            self.user_training = SVD()
            train = self.rating_ds.build_full_trainset()
            self.user_training.fit(train)

    # Establece KNN como método de entrenamiento y predicción de ratings.
    def set_knn_user_training(self):
        if self.user_training_type != 'knn':
            self.user_training_type = 'knn'
            print("Training KNN...")
            self.user_training = KNNBasic(k=80, min_k=20, verbose=True)
            train = self.rating_ds.build_full_trainset()
            self.user_training.fit(train)

    def get_collaborative_recommendations_by_index(self, userId, top=10, ratings=None, user_training=None):
        """
        Obtención de recomendaciones colaborativas por índices.

        Parameters
        ----------

        userId: usuario sobre el que realizar las recomendaciones.

        top: Número de películas a recomendar.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        user_training: Algoritmo de entrenamiento y predicción (por defecto se utiliza el entrenado sobre todas las valoraciones). Especificar para validación cruzada.

        Returns
        -------

        Índices de las películas recomendadas, junto con las estimaciones de las valoraciones.
        """
        if ratings is None:
            ratings = self.ratings
        if user_training is None:
            user_training = self.user_training
        if user_training is None:
            raise ValueError("A training-prediction method must be defined.")
        # nos quedamos unicamente con las películas que no haya visto userId.
        # tomamos unicamente aquellas peliculas que no ha visto
        condition = ratings[ratings['userId'] == userId]
        movieIdSeen = condition['movieId'].unique()
        allmovieId = ratings['movieId'].unique()
        movieIds = np.setdiff1d(allmovieId, movieIdSeen)
        ids = self.di_map.loc[movieIds]['id']
        movie_indices = self.meta_map.loc[ids]

        # buscamos en el dataset con todos los datos las peliculas cuyo Id corresponde con el de la condición anterior y obtenemos los datos deseados
        movies = self.metadata.loc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]
        movies = movies.dropna()
        movies['estimation'] = movies['id'].apply(lambda x: user_training.predict(userId, self.indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('estimation', ascending=False)
        return list(movies.index)[:top], movies['estimation'].head(top)

    def get_collaborative_recommendations(self, userId, top=10, ratings=None, user_training=None):
        """
        Obtención de recomendaciones colaborativas.

        Parameters
        ----------

        userId: usuario sobre el que realizar las recomendaciones.

        top: Número de películas a recomendar.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        user_training: Algoritmo de entrenamiento y predicción (por defecto se utiliza el entrenado sobre todas las valoraciones). Especificar para validación cruzada.


        Returns
        -------

        Dataframe con los títulos de las recomendaciones y las correspondientes estimaciones de las valoraciones.
        """
        movie_indices, estimations = self.get_collaborative_recommendations_by_index(userId, top, ratings, user_training)
        results = pd.concat([self.metadata['title'].iloc[movie_indices], estimations], axis=1)

        return results

    def get_collaborative_recommendations_for_user_by_index(self, userId, top=10, positiveThresh=4.0, ratings=None, user_training=None):
        """
        Obtención de recomendaciones colaborativas por índices (envolvente para validación cruzada).

        Parameters
        ----------

        userId: usuario sobre el que realizar las recomendaciones.

        top: Número de películas a recomendar.

        positiveThresh: ignored.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        user_training: Algoritmo de entrenamiento y predicción (por defecto se utiliza el entrenado sobre todas las valoraciones). Especificar para validación cruzada.

        Returns
        -------

        Índices de las películas recomendadas, junto con las estimaciones de las valoraciones.
        """
        try:
            return self.get_collaborative_recommendations_by_index(userId, top, ratings, user_training)
        except:
            print("No positive recommendations for this user under this threshold.")
            return pd.DataFrame([], columns=['title', 'similarity'])

    def get_collaborative_recommendations_for_user(self, userId, top=10, positiveThresh=4.0, ratings=None, user_training=None):
        """
        Obtención de recomendaciones colaborativas (envolvente para validación cruzada).

        Parameters
        ----------

        userId: usuario sobre el que realizar las recomendaciones.

        top: Número de películas a recomendar.

        positiveThresh: ignored.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        user_training: Algoritmo de entrenamiento y predicción (por defecto se utiliza el entrenado sobre todas las valoraciones). Especificar para validación cruzada.


        Returns
        -------

        Dataframe con los títulos de las recomendaciones y las correspondientes estimaciones de las valoraciones.
        """
        try:
            return self.get_collaborative_recommendations(userId, top, ratings, user_training)
        except:
            print("No positive recommendations for this user under this threshold.")
            return pd.DataFrame([], columns=['title', 'similarity'])

    def get_hybrid_cascade_recommendations_by_index(self, userId, index, top=10, content_top=25, user_training=None):
        """
        Obtención de recomendaciones híbridas en cascada por índices.

        Parameters
        ----------

        userId: usuario sobre el que realizar las recomendaciones.

        index: Entero o lista con los índices de películas sobre los que buscar por contenido.

        top: Número de películas a recomendar.

        content_top: Número de películas a recuperar por contenido en el paso inicial.

        user_training: Algoritmo de entrenamiento y predicción (por defecto se utiliza el entrenado sobre todas las valoraciones). Especificar para validación cruzada.

        Returns
        -------

        Índices de las películas recomendadas, junto con las estimaciones de las valoraciones.
        """
        if self.similarity is None:
            raise ValueError("A similarity metric must be defined.")
        if isinstance(index, int):
            index = [index]
        if user_training is None:
            user_training = self.user_training
        # nmovies = len(index)
        # para el hibrido necesitamos saber el indice de la pelicula en metadata original
        # tras ello, buscamos las peliculas con mayor similitud según el coseno
        # devolvemos las peliculas con mejor estimacion de puntuacion usando svd

        # tmdbId = id_map.loc[clean_title]['id']
        # movie_id = id_map.loc[clean_title]['movieId']
        if index:
            sim_scores = np.array(list(enumerate(np.asarray(self.similarity[index].tocsc().max(axis=0).todense()).ravel())))
            sim_scores[index, 1] = -1.0
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[:content_top]

            movie_indices = [i[0] for i in sim_scores]
            movies = self.metadata.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]

            movies['estimation'] = movies['id'].apply(lambda x: user_training.predict(userId, self.indices_map.loc[x]['movieId']).est)
            movies = movies.sort_values('estimation', ascending=False)
            return list(movies.index)[:top], movies['estimation'].head(top)
        else:
            return [], []

    def get_hybrid_cascade_recommendations(self, userId, title, top=10, content_top=25, user_training=None):
        """
        Obtención de recomendaciones híbridas en cascada.

        Parameters
        ----------

        userId: usuario sobre el que realizar las recomendaciones.

        title: String o lista con los títulos de películas sobre los que buscar por contenido.

        top: Número de películas a recomendar.

        content_top: Número de películas a recuperar por contenido en el paso inicial.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        user_training: Algoritmo de entrenamiento y predicción (por defecto se utiliza el entrenado sobre todas las valoraciones). Especificar para validación cruzada.


        Returns
        -------

        Dataframe con los títulos de las recomendaciones y las correspondientes estimaciones de las valoraciones.
        """
        if self.similarity is None:
            raise ValueError("A similarity metric must be defined.")
        if isinstance(title, str):
            title = [title]

        clean_title = [self._clean_title(t) for t in title]
        # Get the index of the movie that matches the title
        idx = list(self.indices[clean_title])

        movie_indices, estimations = self.get_hybrid_cascade_recommendations_by_index(userId, idx, top, content_top, user_training)
        results = pd.concat([self.metadata['title'].iloc[movie_indices], estimations], axis=1)

        return results

    def get_hybrid_cascade_recommendations_for_user_by_index(self, userId, top=10, content_top=25, positiveThresh=4.0, ratings=None, user_training=None):
        """
        Obtención de recomendaciones híbridas en cascada por índices (envolvente para validación cruzada).

        Parameters
        ----------

        userId: usuario sobre el que realizar las recomendaciones.

        top: Número de películas a recomendar.

        content_top: Número de películas a recuperar por contenido en el paso inicial.

        positiveThresh: Umbral de valoración sobre el que se considera una película relevante.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        user_training: Algoritmo de entrenamiento y predicción (por defecto se utiliza el entrenado sobre todas las valoraciones). Especificar para validación cruzada.

        Returns
        -------

        Índices de las películas recomendadas, junto con las estimaciones de las valoraciones.
        """
        try:
            return self.get_hybrid_cascade_recommendations_by_index(userId, self.get_liked_movies_by_index(userId, positiveThresh, ratings)[0], top, content_top, user_training)
        except:
            print("No positive recommendations for this user under this threshold.")
            return pd.DataFrame([], columns=['title', 'similarity'])

    def get_hybrid_cascade_recommendations_for_user(self, userId, top=10, content_top=25, positiveThresh=4.0, ratings=None, user_training=None):
        """
        Obtención de recomendaciones híbridas en cascada (envolvente para la validación cruzada).

        Parameters
        ----------

        userId: usuario sobre el que realizar las recomendaciones.

        title: String o lista con los títulos de películas sobre los que buscar por contenido.

        top: Número de películas a recomendar.

        content_top: Número de películas a recuperar por contenido en el paso inicial.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        user_training: Algoritmo de entrenamiento y predicción (por defecto se utiliza el entrenado sobre todas las valoraciones). Especificar para validación cruzada.


        Returns
        -------

        Dataframe con los títulos de las recomendaciones y las correspondientes estimaciones de las valoraciones.
        """
        try:
            return self.get_hybrid_cascade_recommendations(userId, self.get_liked_movies(userId, positiveThresh, ratings)['title'], top, content_top, user_training)
        except:
            print("No positive recommendations for this user under this threshold.")
            return pd.DataFrame([], columns=['title', 'similarity'])

    def get_hybrid_weighted_recommendations_by_index(self, userId, index, top=10, ratings=None, user_training=None):
        """
        Obtención de recomendaciones híbridas ponderadas por índices.

        Parameters
        ----------

        userId: usuario sobre el que realizar las recomendaciones.

        index: Entero o lista con los índices de películas sobre los que buscar por contenido.

        top: Número de películas a recomendar.

        user_training: Algoritmo de entrenamiento y predicción (por defecto se utiliza el entrenado sobre todas las valoraciones). Especificar para validación cruzada.

        Returns
        -------

        Índices de las películas recomendadas, junto con los correspondientes score híbridos.
        """
        if self.similarity is None:
            raise ValueError("A similarity metric must be defined.")
        if isinstance(index, int):
            index = [index]
        if user_training is None:
            user_training = self.user_training
        if ratings is None:
            ratings = self.ratings

        if index:
            # Score content-based
            sim_scores = np.array(list(enumerate(np.asarray(self.similarity[index].tocsc().max(axis=0).todense()).ravel())))
            sim_scores[index, 1] = -100.0
            sim_scores = pd.DataFrame(sim_scores, columns=['index', 'sim_score'])

            # nos quedamos unicamente con las películas que no haya visto userId.
            # tomamos unicamente aquellas peliculas que no ha visto
            condition = ratings[ratings['userId'] == userId]
            movieIdSeen = condition['movieId'].unique()
            allmovieId = ratings['movieId'].unique()
            movieIds = np.setdiff1d(allmovieId, movieIdSeen)
            ids = self.di_map.loc[movieIds]['id']
            movie_indices = movie_indices = self.meta_map.loc[ids]
            # buscamos en el dataset con todos los datos las peliculas cuyo Id corresponde con el de la condición anterior y obtenemos los datos deseados
            movies = self.metadata.loc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]
            # movies = movies.dropna()
            movies['estimation'] = movies['id'].apply(lambda x: user_training.predict(userId, self.indices_map.loc[x]['movieId']).est)
            movies = movies.merge(sim_scores, left_index=True, right_on='index')
            movies['estimation'] /= 5

            counts = ratings.groupby('userId').count()['movieId']
            alpha = 0.1
            lmbda = alpha + (1 - 2 * alpha) * sum(counts.loc[userId] >= counts) / counts.count()
            movies['hybrid_score'] = (1 - lmbda) * movies['sim_score'] + lmbda * movies['estimation']
            movies = movies.sort_values('hybrid_score', ascending=False)

            return list(movies.index)[:top], movies['hybrid_score'].head(top)
        else:
            return [], []

    def get_hybrid_weighted_recommendations(self, userId, title, top=10, ratings=None, user_training=None):
        """
        Obtención de recomendaciones híbridas ponderadas.

        Parameters
        ----------

        userId: usuario sobre el que realizar las recomendaciones.

        title: String o lista con los títulos de películas sobre los que buscar por contenido.

        top: Número de películas a recomendar.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        user_training: Algoritmo de entrenamiento y predicción (por defecto se utiliza el entrenado sobre todas las valoraciones). Especificar para validación cruzada.


        Returns
        -------

        Dataframe con los títulos de las recomendaciones y los correspondientes score híbridos.
        """
        if self.similarity is None:
            raise ValueError("A similarity metric must be defined.")
        if isinstance(title, str):
            title = [title]

        clean_title = [self._clean_title(t) for t in title]
        # Get the index of the movie that matches the title
        idx = list(self.indices[clean_title])

        movie_indices, scores = self.get_hybrid_weighted_recommendations_by_index(userId, idx, top, ratings, user_training)
        results = pd.concat([self.metadata['title'].iloc[movie_indices], scores], axis=1)

        return results

    def get_hybrid_weighted_recommendations_for_user_by_index(self, userId, top=10, positiveThresh=4.0, ratings=None, user_training=None):
        """
        Obtención de recomendaciones híbridas ponderadas por índices (envolvente para validación cruzada).

        Parameters
        ----------

        userId: usuario sobre el que realizar las recomendaciones.

        top: Número de películas a recomendar.

        positiveThresh: Umbral de valoración sobre el que se considera una película relevante.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        user_training: Algoritmo de entrenamiento y predicción (por defecto se utiliza el entrenado sobre todas las valoraciones). Especificar para validación cruzada.

        Returns
        -------

        Índices de las películas recomendadas, junto con los score híbridos.
        """
        try:
            return self.get_hybrid_weighted_recommendations_by_index(userId, self.get_liked_movies_by_index(userId, positiveThresh, ratings)[0], top, ratings, user_training)
        except:
            print("No positive recommendations for this user under this threshold.")
            return pd.DataFrame([], columns=['title', 'similarity'])

    def get_hybrid_weighted_recommendations_for_user(self, userId, top=10, positiveThresh=4.0, ratings=None, user_training=None):
        """
        Obtención de recomendaciones híbridas en cascada (envolvente para la validación cruzada).

        Parameters
        ----------

        userId: usuario sobre el que realizar las recomendaciones.

        title: String o lista con los títulos de películas sobre los que buscar por contenido.

        top: Número de películas a recomendar.

        content_top: Número de películas a recuperar por contenido en el paso inicial.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        user_training: Algoritmo de entrenamiento y predicción (por defecto se utiliza el entrenado sobre todas las valoraciones). Especificar para validación cruzada.


        Returns
        -------

        Dataframe con los títulos de las recomendaciones y los correspondientes score híbridos.
        """
        try:
            return self.get_hybrid_weighted_recommendations(userId, self.get_liked_movies(userId, positiveThresh, ratings)['title'], top, ratings, user_training)
        except:
            print("No positive recommendations for this user under this threshold.")
            return pd.DataFrame([], columns=['title', 'similarity'])

    def get_watched_movies_by_index(self, userId, ratings=None):
        """
        Obtención de películas vistas por un usuario (índices).

        Parameters
        ----------

        userId: usuario sobre el que buscar.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        Returns
        -------

        Índices de las películas vistas, junto con las valoraciones realizadas por el usuario.
        """
        if ratings is None:
            ratings = self.ratings
        watched_mid = self.ratings[self.ratings['userId'] == userId]
        watched_id = self.di_map.loc[watched_mid['movieId']]['id']
        watched_indices = self.meta_map.loc[watched_id]
        ratings = watched_mid['rating']
        return list(watched_indices), ratings

    def get_watched_movies(self, userId, user_ratings=None):
        """
        Obtención de películas vistas por un usuario.

        Parameters
        ----------

        userId: usuario sobre el que buscar.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        Returns
        -------

        DataFrame con los títulos de las películas vistas, y sus correspondientes valoraciones.
        """
        indices, ratings = self.get_watched_movies_by_index(userId, user_ratings)
        results = pd.DataFrame(list(zip(self.metadata['title'].iloc[indices], ratings)), columns=['title', 'rating'])
        return results

    def get_liked_movies_by_index(self, userId, positiveThresh=4.0, ratings=None):
        """
        Obtención de películas que han gustado a un usuario (índices).

        Parameters
        ----------

        userId: usuario sobre el que buscar.

        positiveThresh: umbral a partir del cual se considera que una película ha gustado.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        Returns
        -------

        Índices de las películas que han gustado al usuario, junto con las valoraciones realizadas por el usuario.
        """
        if ratings is None:
            ratings = self.ratings
        liked = ratings[ratings['rating'] >= positiveThresh]
        watched_mid = liked[liked['userId'] == userId]
        watched_id = self.di_map.loc[watched_mid['movieId']]['id']
        watched_indices = self.meta_map.loc[watched_id]
        ratings = watched_mid['rating']
        return list(watched_indices), ratings

    def get_liked_movies(self, userId, positiveThresh=4.0, user_ratings=None):
        """
        Obtención de películas vistas por un usuario.

        Parameters
        ----------

        userId: usuario sobre el que buscar.

        positiveThresh: umbral a partir del cual se considera que una película ha gustado.

        ratings: Dataframe de valoraciones (por defecto se usan todas las valoraciones). Especificar para validación cruzada.

        Returns
        -------

        DataFrame con los títulos de las películas que han gustado al usuario, y sus correspondientes valoraciones.
        """
        indices, ratings = self.get_liked_movies_by_index(userId, positiveThresh, user_ratings)
        results = pd.DataFrame(list(zip(self.metadata['title'].iloc[indices], ratings)), columns=['title', 'rating'])
        return results

    def _evaluate_recommendations(self, rec_list):
        """
        Función que obtiene medidas de evaluación a partir de la lista de recomendaciones sobre todos los usuarios.

        Parameters
        ----------

        rec_list : Lista con las listas de recomendaciones (en índices) de todos los usuarios.

        Returns
        -------

        Diccionario con las medidas coverage, personalization, intralist_overview_similarity, intralist_cgk_similarity y novelty, y los correspondientes valores.
        """
        print("Preprocessing...")
        movie_indices = list(self.metadata.index)

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.metadata['overview'])

        features = ['cast', 'keywords', 'genres']
        top_features = pd.DataFrame()
        for feature in features:
            top_features['top_' + feature] = self.metadata['clean_' + feature].apply(lambda x: x[:3])
        top_features['clean_director'] = self.metadata['clean_director']
        top_features['soup'] = top_features.apply(self._create_soup, axis=1)

        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(top_features['soup'])

        print("Calculating coverage...")
        cov = coverage(rec_list, movie_indices)
        print("Calculating personalization...")
        pers = personalization(rec_list)
        print("Calculating intra-list overview similarity...")
        ils_overview = intra_list_similarity(rec_list, tfidf_matrix)
        print("Calculating intra-list CGK similarity...")
        ils_cgk = intra_list_similarity(rec_list, count_matrix)
        print("Calculating novelty...")
        nov = novelty(rec_list)

        return {'coverage': cov,
                'personalization': pers,
                'intralist_overview_similarity': ils_overview,
                'intralist_cgk_similarity': ils_cgk,
                'novelty': nov}

    def evaluate_popularity_recommendations(self, top=10, positiveThresh=4.0):
        """
        Función que obtiene las medidas de evaluación no supervisadas para las recomendaciones por popularidad.

        Parameters
        ----------

        top: Tamaño de las listas de recomendación.

        positiveThresh: Ignored.


        Returns
        -------

        Diccionario con las medidas coverage, personalization, intralist_overview_similarity, intralist_cgk_similarity y novelty, y los correspondientes valores.
        """
        print("Obtaining recommendations for every user (this may take a while)...")
        user_ids = self.ratings['userId'].unique()
        nusers = len(user_ids)

        rec_list = [print(str(uid) + " / " + str(nusers) + "\r", end='\r') or self.get_popularity_recommendations_for_user_by_index(uid, top, positiveThresh)[0] for uid in user_ids]

        return self._evaluate_recommendations(rec_list)

    def evaluate_content_recommendations(self, top=10, positiveThresh=4.0):
        """
        Función que obtiene las medidas de evaluación no supervisadas para las recomendaciones por contenido.

        Parameters
        ----------

        top: Tamaño de las listas de recomendación.

        positiveThresh: umbral de valoración para el que se considera que una película es relevante.


        Returns
        -------

        Diccionario con las medidas coverage, personalization, intralist_overview_similarity, intralist_cgk_similarity y novelty, y los correspondientes valores.
        """
        print("Obtaining recommendations for every user (this may take a while)...")
        user_ids = self.ratings['userId'].unique()
        nusers = len(user_ids)

        rec_list = [print(str(uid) + " / " + str(nusers) + "\r", end='\r') or self.get_content_recommendations_for_user_by_index(uid, top, positiveThresh)[0] for uid in user_ids]

        return self._evaluate_recommendations(rec_list)

    def evaluate_collaborative_recommendations(self, top=10, positiveThresh=4.0):
        """
        Función que obtiene las medidas de evaluación no supervisadas para las recomendaciones colaborativas.

        Parameters
        ----------

        top: Tamaño de las listas de recomendación.

        positiveThresh: Ignored.


        Returns
        -------

        Diccionario con las medidas coverage, personalization, intralist_overview_similarity, intralist_cgk_similarity y novelty, y los correspondientes valores.
        """
        print("Obtaining recommendations for every user (this may take a while)...")
        user_ids = self.ratings['userId'].unique()
        nusers = len(user_ids)

        rec_list = [print(str(uid) + " / " + str(nusers) + "\r", end='\r') or self.get_collaborative_recommendations_for_user_by_index(uid, top, positiveThresh)[0] for uid in user_ids]

        return self._evaluate_recommendations(rec_list)

    def evaluate_hybrid_cascade_recommendations(self, top=10, content_top=25, positiveThresh=4.0):
        """
        Función que obtiene las medidas de evaluación no supervisadas para las recomendaciones híbridas en cascada.

        Parameters
        ----------

        top: Tamaño de las listas de recomendación.

        content_top: Número de películas a recuperar por contenido en la fase inicial.

        positiveThresh: umbral de valoración para el que se considera que una película es relevante.


        Returns
        -------

        Diccionario con las medidas coverage, personalization, intralist_overview_similarity, intralist_cgk_similarity y novelty, y los correspondientes valores.
        """
        print("Obtaining recommendations for every user (this may take a while)...")
        user_ids = self.ratings['userId'].unique()
        nusers = len(user_ids)

        rec_list = [print(str(uid) + " / " + str(nusers) + "\r", end='\r') or self.get_hybrid_cascade_recommendations_for_user_by_index(uid, top, content_top, positiveThresh)[0] for uid in user_ids]

        return self._evaluate_recommendations(rec_list)

    def evaluate_hybrid_weighted_recommendations(self, top=10, positiveThresh=4.0):
        """
        Función que obtiene las medidas de evaluación no supervisadas para las recomendaciones híbridas ponderadas.

        Parameters
        ----------

        top: Tamaño de las listas de recomendación.

        positiveThresh: umbral de valoración para el que se considera que una película es relevante.


        Returns
        -------

        Diccionario con las medidas coverage, personalization, intralist_overview_similarity, intralist_cgk_similarity y novelty, y los correspondientes valores.
        """
        print("Obtaining recommendations for every user (this may take a while)...")
        user_ids = self.ratings['userId'].unique()
        nusers = len(user_ids)

        rec_list = [print(str(uid) + " / " + str(nusers) + "\r", end='\r') or self.get_hybrid_weighted_recommendations_for_user_by_index(uid, top, positiveThresh)[0] for uid in user_ids]

        return self._evaluate_recommendations(rec_list)

    def _generate_train_test(self, train_indices, test_indices, movie_train=0.8):
        """
        Generación de conjuntos para la validación cruzada.

        Parameters
        ----------

        train_indices: Índices de la partición de entrenamiento.

        test_indices: Índices de la partición de test.

        movie_train: Fracción dedicada a elaborar las recomendaciones del conjunto test.

        Returns
        -------

        Conjuntos de usuarios para entrenamiento, para generación de recomendaciones en test y para evaluación de las recomendaciones en test.
        """
        ratings_train = self.ratings.loc[self.ratings.index[train_indices]]
        ratings_test = self.ratings.loc[self.ratings.index[test_indices]]

        # Dividimos test en peliculas ya vistas y películas posteriores (usadas para validar)
        div = ratings_test.groupby('userId')[['userId', 'timestamp']].quantile(movie_train)
        test_pre = ratings_test[ratings_test['timestamp'] < div.loc[ratings_test['userId']]['timestamp'].values]
        test_pos = ratings_test[ratings_test['timestamp'] >= div.loc[ratings_test['userId']]['timestamp'].values]
        return ratings_train, test_pre, test_pos

    def _partitionate(self, n_folds=5, movie_train=0.8, random_state=28):
        """
        Generación de las particiones de validación cruzada. Se almacenan en el atributo de instancia 'partitions'.

        Parameters
        ----------

        n_folds: Número de particiones a realizar.

        movie_train: Fracción dedicada a elaborar las recomendaciones del conjunto test.

        random_state: Semilla para generar las particiones.
        """
        print("Partitioning...")
        kf = KFold(n_folds, random_state=random_state)
        self.partitions = [self._generate_train_test(train, test, movie_train) for train, test in kf.split(self.ratings)]

    def validate_popularity_recommendations(self, n_folds=5, movie_train=0.8, top=10, positiveThresh=4.0, random_state=28):
        """
        Validación cruzada de las recomendaciones por popularidad.

        Parameters
        ----------

        n_folds: Número de particiones a realizar.

        movie_train: Fracción dedicada a elaborar las recomendaciones del conjunto test.

        top: Tamaño de las listas de recomendación.

        positiveThresh: umbral de valoración a partir del cual una película se considera relevante.

        random_state: Semilla para generar las particiones.
        """
        self._partitionate(n_folds, movie_train, random_state)
        prec = np.empty([n_folds, top])
        rec = np.empty([n_folds, top])
        prec_w = np.empty([n_folds, top])
        for i, (train, test_pre, test_pos) in enumerate(self.partitions):
            print("Validating Fold " + str(i + 1) + "...")
            print("Obtaining recommendations for every user (this may take a while)...")
            user_ids = test_pre['userId'].unique()
            nusers = len(user_ids)
            rec_preds = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_popularity_recommendations_for_user_by_index(uid, top, positiveThresh, test_pre)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (1/3)...")
            rec_real = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_liked_movies_by_index(uid, positiveThresh, test_pos)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (2/3)...")
            rec_watched = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_watched_movies_by_index(uid, test_pos)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (3/3)...")
            rec_preds_watched = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or np.intersect1d(rec_preds[j], rec_watched[j]) for j, uid in enumerate(user_ids)]
            # print("RECALL: ", recall(rec_preds, rec_real))
            # print("PRECISION: ", precision(rec_preds, rec_real))
            print("Evaluating...")
            for k in range(1, top + 1):
                prec[i, k - 1] = precision(rec_real, rec_preds, k=k)
                rec[i, k - 1] = recall(rec_real, rec_preds, k=k)
                prec_w[i, k - 1] = precision(rec_real, rec_preds_watched, k=k)
                # print("* P@" + str(k), precision(rec_real, rec_preds, k=k))
                # print("* R@" + str(k), recall(rec_real, rec_preds, k=k))
                # print("* P|w@" + str(k), precision(rec_real, rec_preds_watched, k=k))
                # print("* R|w@" + str(k), recall(rec_real, rec_preds_watched, k=k))  # R|w = R

        prec = pd.DataFrame(prec, columns=range(1, top + 1))
        rec = pd.DataFrame(rec, columns=range(1, top + 1))
        prec_w = pd.DataFrame(prec_w, columns=range(1, top + 1))

        return {'precision': prec.mean(),
                'recall': rec.mean(),
                'precision_watched': prec_w.mean()}

    def validate_content_recommendations(self, n_folds=5, movie_train=0.8, top=10, positiveThresh=4.0, random_state=28):
        """
        Validación cruzada de las recomendaciones por contenido.

        Parameters
        ----------

        n_folds: Número de particiones a realizar.

        movie_train: Fracción dedicada a elaborar las recomendaciones del conjunto test.

        top: Tamaño de las listas de recomendación.

        positiveThresh: umbral de valoración a partir del cual una película se considera relevante.

        random_state: Semilla para generar las particiones.
        """
        self._partitionate(n_folds, movie_train, random_state)
        prec = np.empty([n_folds, top])
        rec = np.empty([n_folds, top])
        prec_w = np.empty([n_folds, top])
        for i, (train, test_pre, test_pos) in enumerate(self.partitions):
            print("Validating Fold " + str(i + 1) + "...")
            print("Obtaining recommendations for every user (this may take a while)...")
            user_ids = test_pre['userId'].unique()
            nusers = len(user_ids)
            rec_preds = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_content_recommendations_for_user_by_index(uid, top, positiveThresh, test_pre)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (1/3)...")
            rec_real = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_liked_movies_by_index(uid, positiveThresh, test_pos)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (2/3)...")
            rec_watched = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_watched_movies_by_index(uid, test_pos)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (3/3)...")
            rec_preds_watched = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or np.intersect1d(rec_preds[j], rec_watched[j]) for j, uid in enumerate(user_ids)]
            # print("RECALL: ", recall(rec_preds, rec_real))
            # print("PRECISION: ", precision(rec_preds, rec_real))
            print("Evaluating...")
            for k in range(1, top + 1):
                prec[i, k - 1] = precision(rec_real, rec_preds, k=k)
                rec[i, k - 1] = recall(rec_real, rec_preds, k=k)
                prec_w[i, k - 1] = precision(rec_real, rec_preds_watched, k=k)
                # print("* P@" + str(k), precision(rec_real, rec_preds, k=k))
                # print("* R@" + str(k), recall(rec_real, rec_preds, k=k))
                # print("* P|w@" + str(k), precision(rec_real, rec_preds_watched, k=k))
                # print("* R|w@" + str(k), recall(rec_real, rec_preds_watched, k=k))  # R|w = R

        prec = pd.DataFrame(prec, columns=range(1, top + 1))
        rec = pd.DataFrame(rec, columns=range(1, top + 1))
        prec_w = pd.DataFrame(prec_w, columns=range(1, top + 1))

        return {'precision': prec.mean(),
                'recall': rec.mean(),
                'precision_watched': prec_w.mean()}

    # Función privada para sustituir funcionalidad de Surprise.
    def _raw_folds_surprise(self, train, test):
        for i in range(1):
            # reader = Reader()
            # train = Dataset.load_from_df(train[['userId', 'movieId', 'rating']], reader)
            train = [(uid, iid, float(r), None) for (uid, iid, r, t) in train.itertuples(index=False)]
            # test = Dataset.load_from_df(test[['userId', 'movieId', 'rating']], reader)
            test = [(uid, iid, float(r), None) for (uid, iid, r, t) in test.itertuples(index=False)]
            yield train, test

    def validate_collaborative_recommendations(self, n_folds=5, movie_train=0.8, top=10, positiveThresh=4.0, random_state=28):
        """
        Validación cruzada de las recomendaciones colaborativas.

        Parameters
        ----------

        n_folds: Número de particiones a realizar.

        movie_train: Fracción dedicada a elaborar las recomendaciones del conjunto test.

        top: Tamaño de las listas de recomendación.

        positiveThresh: umbral de valoración a partir del cual una película se considera relevante.

        random_state: Semilla para generar las particiones.
        """
        self._partitionate(n_folds, movie_train, random_state)
        prec = np.empty([n_folds, top])
        rec = np.empty([n_folds, top])
        prec_w = np.empty([n_folds, top])
        for i, (train, test_pre, test_pos) in enumerate(self.partitions):
            print("Validating Fold " + str(i + 1) + "...")
            print("Training...")
            # Chanchullo para solucionar la falta de funcionalidad de surprise
            df = DatasetUserFolds([('recommender_system.py', 'recommender_system.py')], Reader())
            df.raw_folds = lambda: self._raw_folds_surprise(train, test_pre)

            if self.user_training_type == 'svd':
                alg = SVD()
            elif self.user_training_type == 'knn':
                alg = KNNBasic(80, 20)
            for train_s, test_s in df.folds():
                np.random.seed(random_state)
                random.seed(random_state)
                alg.fit(train_s)

            print("Obtaining recommendations for every user (this may take a while)...")
            user_ids = test_pre['userId'].unique()
            nusers = len(user_ids)
            rec_preds = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_collaborative_recommendations_for_user_by_index(uid, top, positiveThresh, test_pre, alg)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (1/3)...")
            rec_real = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_liked_movies_by_index(uid, positiveThresh, test_pos)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (2/3)...")
            rec_watched = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_watched_movies_by_index(uid, test_pos)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (3/3)...")
            rec_preds_watched = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or np.intersect1d(rec_preds[j], rec_watched[j]) for j, uid in enumerate(user_ids)]
            # print("RECALL: ", recall(rec_preds, rec_real))
            # print("PRECISION: ", precision(rec_preds, rec_real))
            print("Evaluating...")
            for k in range(1, top + 1):
                prec[i, k - 1] = precision(rec_real, rec_preds, k=k)
                rec[i, k - 1] = recall(rec_real, rec_preds, k=k)
                prec_w[i, k - 1] = precision(rec_real, rec_preds_watched, k=k)
                # print("* P@" + str(k), precision(rec_real, rec_preds, k=k))
                # print("* R@" + str(k), recall(rec_real, rec_preds, k=k))
                # print("* P|w@" + str(k), precision(rec_real, rec_preds_watched, k=k))
                # print("* R|w@" + str(k), recall(rec_real, rec_preds_watched, k=k))  # R|w = R

        prec = pd.DataFrame(prec, columns=range(1, top + 1))
        rec = pd.DataFrame(rec, columns=range(1, top + 1))
        prec_w = pd.DataFrame(prec_w, columns=range(1, top + 1))

        return {'precision': prec.mean(),
                'recall': rec.mean(),
                'precision_watched': prec_w.mean()}

    def validate_hybrid_cascade_recommendations(self, n_folds=5, movie_train=0.8, top=10, content_top=25, positiveThresh=4.0, random_state=28):
        """
        Validación cruzada de las recomendaciones híbridas en cascada.

        Parameters
        ----------

        n_folds: Número de particiones a realizar.

        movie_train: Fracción dedicada a elaborar las recomendaciones del conjunto test.

        top: Tamaño de las listas de recomendación.

        content_top: Número de recomendaciones por contenido a recuperar en la fase inicial.

        positiveThresh: umbral de valoración a partir del cual una película se considera relevante.

        random_state: Semilla para generar las particiones.
        """
        self._partitionate(n_folds, movie_train, random_state)
        prec = np.empty([n_folds, top])
        rec = np.empty([n_folds, top])
        prec_w = np.empty([n_folds, top])
        for i, (train, test_pre, test_pos) in enumerate(self.partitions):
            print("Validating Fold " + str(i + 1) + "...")
            print("Training...")
            # Chanchullo para solucionar la falta de funcionalidad de surprise
            df = DatasetUserFolds([('recommender_system.py', 'recommender_system.py')], Reader())
            df.raw_folds = lambda: self._raw_folds_surprise(train, test_pre)
            if self.user_training_type == 'svd':
                alg = SVD()
            elif self.user_training_type == 'knn':
                alg = KNNBasic(80, 20)
            for train_s, test_s in df.folds():
                np.random.seed(random_state)
                random.seed(random_state)
                alg.fit(train_s)

            print("Obtaining recommendations for every user (this may take a while)...")
            user_ids = test_pre['userId'].unique()
            nusers = len(user_ids)
            rec_preds = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_hybrid_cascade_recommendations_for_user_by_index(uid, top, content_top, positiveThresh, test_pre, alg)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (1/3)...")
            rec_real = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_liked_movies_by_index(uid, positiveThresh, test_pos)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (2/3)...")
            rec_watched = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_watched_movies_by_index(uid, test_pos)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (3/3)...")
            rec_preds_watched = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or np.intersect1d(rec_preds[j], rec_watched[j]) for j, uid in enumerate(user_ids)]
            # print("RECALL: ", recall(rec_preds, rec_real))
            # print("PRECISION: ", precision(rec_preds, rec_real))
            print("Evaluating...")
            for k in range(1, top + 1):
                prec[i, k - 1] = precision(rec_real, rec_preds, k=k)
                rec[i, k - 1] = recall(rec_real, rec_preds, k=k)
                prec_w[i, k - 1] = precision(rec_real, rec_preds_watched, k=k)
                # print("* P@" + str(k), precision(rec_real, rec_preds, k=k))
                # print("* R@" + str(k), recall(rec_real, rec_preds, k=k))
                # print("* P|w@" + str(k), precision(rec_real, rec_preds_watched, k=k))
                # print("* R|w@" + str(k), recall(rec_real, rec_preds_watched, k=k))  # R|w = R

        prec = pd.DataFrame(prec, columns=range(1, top + 1))
        rec = pd.DataFrame(rec, columns=range(1, top + 1))
        prec_w = pd.DataFrame(prec_w, columns=range(1, top + 1))

        return {'precision': prec.mean(),
                'recall': rec.mean(),
                'precision_watched': prec_w.mean()}

    def validate_hybrid_weighted_recommendations(self, n_folds=5, movie_train=0.8, top=10, positiveThresh=4.0, random_state=28):
        """
        Validación cruzada de las recomendaciones híbridas ponderadas.

        Parameters
        ----------

        n_folds: Número de particiones a realizar.

        movie_train: Fracción dedicada a elaborar las recomendaciones del conjunto test.

        top: Tamaño de las listas de recomendación.

        positiveThresh: umbral de valoración a partir del cual una película se considera relevante.

        random_state: Semilla para generar las particiones.
        """
        self._partitionate(n_folds, movie_train, random_state)
        prec = np.empty([n_folds, top])
        rec = np.empty([n_folds, top])
        prec_w = np.empty([n_folds, top])
        for i, (train, test_pre, test_pos) in enumerate(self.partitions):
            print("Validating Fold " + str(i + 1) + "...")
            print("Training...")
            # Chanchullo para solucionar la falta de funcionalidad de surprise
            df = DatasetUserFolds([('recommender_system.py', 'recommender_system.py')], Reader())
            df.raw_folds = lambda: self._raw_folds_surprise(train, test_pre)
            if self.user_training_type == 'svd':
                alg = SVD()
            elif self.user_training_type == 'knn':
                alg = KNNBasic(80, 20)
            for train_s, test_s in df.folds():
                np.random.seed(random_state)
                random.seed(random_state)
                alg.fit(train_s)

            print("Obtaining recommendations for every user (this may take a while)...")
            user_ids = test_pre['userId'].unique()
            nusers = len(user_ids)
            rec_preds = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_hybrid_weighted_recommendations_for_user_by_index(uid, top, positiveThresh, test_pre, alg)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (1/3)...")
            rec_real = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_liked_movies_by_index(uid, positiveThresh, test_pos)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (2/3)...")
            rec_watched = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or self.get_watched_movies_by_index(uid, test_pos)[0] for j, uid in enumerate(user_ids)]
            print("Obtaining test values (3/3)...")
            rec_preds_watched = [print(str(j) + " / " + str(nusers) + "\r", end='\r') or np.intersect1d(rec_preds[j], rec_watched[j]) for j, uid in enumerate(user_ids)]
            # print("RECALL: ", recall(rec_preds, rec_real))
            # print("PRECISION: ", precision(rec_preds, rec_real))
            print("Evaluating...")
            for k in range(1, top + 1):
                prec[i, k - 1] = precision(rec_real, rec_preds, k=k)
                rec[i, k - 1] = recall(rec_real, rec_preds, k=k)
                prec_w[i, k - 1] = precision(rec_real, rec_preds_watched, k=k)
                # print("* P@" + str(k), precision(rec_real, rec_preds, k=k))
                # print("* R@" + str(k), recall(rec_real, rec_preds, k=k))
                # print("* P|w@" + str(k), precision(rec_real, rec_preds_watched, k=k))
                # print("* R|w@" + str(k), recall(rec_real, rec_preds_watched, k=k))  # R|w = R

        prec = pd.DataFrame(prec, columns=range(1, top + 1))
        rec = pd.DataFrame(rec, columns=range(1, top + 1))
        prec_w = pd.DataFrame(prec_w, columns=range(1, top + 1))

        return {'precision': prec.mean(),
                'recall': rec.mean(),
                'precision_watched': prec_w.mean()}


# # Ejemplo de uso:
# $ python -i recommender_system.py
#
# > recsys = RecommenderSystem() # Inicializar
#
# # Aproximación basada en contenido
# > recsys.set_XXXX_similarity_metric()  # Elegir métrica de similaridad
# > recsys.get_content_recommendations('The Dark Knight Rises')
#
# # Aproximación colaborativa
# > recsys.set_XXXX_user_training()
# > recsys.get_collaborative_recommendations(1)
#
# # Aproximación híbrida
# > recsys.set_XXXX_similarity_metric()
# > recsys.set_XXXX_user_training()
# > recsys.get_hybrid_cascade_recommendations(1, 'The Dark Knight Rises')

# Use these before fitting an algorithm
# np.random.seed(28)
# random.seed(28)

# Código de inicialización básico
# if __name__ == "__main__":
#     np.random.seed(28)
#     random.seed(28)

#     recsys = RecommenderSystem()

#     recsys.set_overview_similarity_metric()
#     recsys.set_svd_user_training()
