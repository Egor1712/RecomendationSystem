import os.path

import numpy as np
from src.utils import load_model, load_preprocessor
import joblib


class HmRecommender:
    def __init__(self, model_type):
        model_path = os.path.join(model_type, 'output', 'model_best')
        mappings_path = os.path.join(model_type, 'output', 'mappings.pkl')
        self.model = load_model(model_path)
        self.mappings = joblib.load(mappings_path)
        self.user_to_idx = self.mappings['user_to_idx']
        self.item_to_idx = self.mappings['item_to_idx']
        self.user_ids = self.mappings['user_ids']
        self.item_ids = self.mappings['item_ids']
        self.user_features = None
        self.item_features = None
        self.popular_items = self.item_ids[:100]  # временно, будет переопределён


    def set_features(self, user_features, item_features):
        self.user_features = user_features
        self.item_features = item_features

    def recommend(self, user_id, top_k=12):
        if user_id not in self.user_to_idx:
            # Если пользователь новый, возвращаем популярные товары
            return self._get_popular_items(top_k)

        user_idx = self.user_to_idx[user_id]
        # Получаем оценки для всех товаров
        scores = self.model.predict(user_idx, np.arange(len(self.item_ids)),
                                    user_features=self.user_features,
                                    item_features=self.item_features)
        # Если нужно исключить уже купленные товары – загружаем историю (упрощённо: из interaction)
        # В этом прототипе пропустим для простоты, но можно добавить.
        top_indices = np.argsort(-scores)[:top_k]
        recommendations = [(self.item_ids[i], float(scores[i])) for i in top_indices]
        return recommendations

    def _get_popular_items(self, top_k):
        return [(item_id, 0.0) for item_id in self.popular_items[:top_k]]