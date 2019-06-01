from recommender_system import RecommenderSystem
import numpy as np
import random


np.random.seed(28)
random.seed(28)

recsys = RecommenderSystem()

np.random.seed(28)
random.seed(28)
recsys.set_overview_similarity_metric()

np.random.seed(28)
random.seed(28)
recsys.set_svd_user_training()

# Evaluación de popularity-based
print("Popularity-based")
print(recsys.evaluate_popularity_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_popularity_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))

# Evaluación de content-based con overview similarity
print("Content-based con overview similarity")
print(recsys.evaluate_content_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_content_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))

# Evaluación de user-based con SVD
print("User-based con SVD")
print(recsys.evaluate_collaborative_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_collaborative_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))

# Evaluación de híbrido con overview similarity y SVD
print("Híbrido con overview similarity y SVD")
print(recsys.evaluate_hybrid_cascade_recommendations(top=10, content_top=25, positiveThresh=3.5))
print(recsys.validate_hybrid_cascade_recommendations(n_folds=5, movie_train=0.5, top=20, content_top=25, positiveThresh=3.5, random_state=28))
print(recsys.evaluate_hybrid_weighted_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_hybrid_weighted_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))

np.random.seed(28)
random.seed(28)
recsys.set_cgk_similarity_metric()

# Evaluación de content-based con cgk similarity
print("Content-based con cgk similarity")
print(recsys.evaluate_content_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_content_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))

# Evaluación de híbrido con cgk similarity y SVD
print("Híbrido con cgk similarity y SVD")
print(recsys.evaluate_hybrid_cascade_recommendations(top=10, content_top=25, positiveThresh=3.5))
print(recsys.validate_hybrid_cascade_recommendations(n_folds=5, movie_train=0.5, top=20, content_top=25, positiveThresh=3.5, random_state=28))
print(recsys.evaluate_hybrid_weighted_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_hybrid_weighted_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))

np.random.seed(28)
random.seed(28)
recsys.set_knn_user_training()

# Evaluación de user-based con KNN
print("User-based con KNN")
print(recsys.evaluate_collaborative_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_collaborative_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))

# Evaluación de híbrido con cgk similarity y KNN
print("Híbrido con cgk similarity y KNN")
print(recsys.evaluate_hybrid_cascade_recommendations(top=10, content_top=25, positiveThresh=3.5))
print(recsys.validate_hybrid_cascade_recommendations(n_folds=5, movie_train=0.5, top=20, content_top=25, positiveThresh=3.5, random_state=28))
print(recsys.evaluate_hybrid_weighted_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_hybrid_weighted_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))


np.random.seed(28)
random.seed(28)
recsys.set_overview_similarity_metric()

# Evaluación de híbrido con overview similarity y KNN
print("Híbrido con overview similarity y KNN")
print(recsys.evaluate_hybrid_cascade_recommendations(top=10, content_top=25, positiveThresh=3.5))
print(recsys.validate_hybrid_cascade_recommendations(n_folds=5, movie_train=0.5, top=20, content_top=25, positiveThresh=3.5, random_state=28))
print(recsys.evaluate_hybrid_weighted_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_hybrid_weighted_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))
