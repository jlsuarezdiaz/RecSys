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

# Evaluación de content-based con overview similarity
print(recsys.evaluate_content_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_content_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))

# Evaluación de user-based con SVD
print(recsys.evaluate_collaborative_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_collaborative_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))

# Evaluación de híbrido con overview similarity y SVD
print(recsys.evaluate_hybrid_recommendations(top=10, content_top=25, positiveThresh=3.5))
print(recsys.validate_hybrid_recommendations(n_folds=5, movie_train=0.5, top=20, content_top=25, positiveThresh=3.5, random_state=28))

np.random.seed(28)
random.seed(28)
recsys.set_cgk_similarity_metric()

# Evaluación de content-based con cgk similarity
print(recsys.evaluate_content_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_content_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))

# Evaluación de híbrido con cgk similarity y SVD
print(recsys.evaluate_hybrid_recommendations(top=10, content_top=25, positiveThresh=3.5))
print(recsys.validate_hybrid_recommendations(n_folds=5, movie_train=0.5, top=20, content_top=25, positiveThresh=3.5, random_state=28))

np.random.seed(28)
random.seed(28)
recsys.set_knn_user_training()

# Evaluación de user-based con KNN
print(recsys.evaluate_collaborative_recommendations(top=10, positiveThresh=3.5))
print(recsys.validate_collaborative_recommendations(n_folds=5, movie_train=0.5, top=20, positiveThresh=3.5, random_state=28))

# Evaluación de híbrido con cgk similarity y KNN
print(recsys.evaluate_hybrid_recommendations(top=10, content_top=25, positiveThresh=3.5))
print(recsys.validate_hybrid_recommendations(n_folds=5, movie_train=0.5, top=20, content_top=25, positiveThresh=3.5, random_state=28))
