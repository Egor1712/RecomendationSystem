from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from src.inference import HmRecommender
from src.preprocess import load_hm_data, build_user_item_features

app = FastAPI(title="H&M Recommendation API")

# Монтируем папки для статики и изображений
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")


# Глобальные объекты
recommender = None
articles_df = None
transactions_df = None
user_to_idx = None


class ProductInfo(BaseModel):
    article_id: int
    product_name: Optional[str] = None
    image_url: str
    score: Optional[float] = None


class HistoryResponse(BaseModel):
    user_id: str
    purchases: List[ProductInfo]


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[ProductInfo]


@app.on_event("startup")
async def startup_event():
    global recommender, articles_df, transactions_df, user_to_idx
    print("Загрузка данных...")
    articles_df, customers_df, transactions_df = load_hm_data()

    # Строим признаки и обучаем/загружаем модель (упрощённо – загружаем готовую)
    interaction, user_features, item_features, user_to_idx_map, item_to_idx_map = build_user_item_features(
        articles_df, customers_df, transactions_df
    )
    user_to_idx = user_to_idx_map

    # Инициализируем рекомендателя
    recommender = HmRecommender()
    recommender.set_features(user_features, item_features)

    print("API готов")


def get_product_info(article_id: int) -> ProductInfo:
    """Возвращает информацию о товаре для отображения."""
    row = articles_df[articles_df['article_id'] == article_id]
    if len(row) == 0:
        return ProductInfo(
            article_id=article_id,
            product_name="Unknown",
            image_url=f"/images/0{article_id}.jpg"
        )
    name = row.iloc[0].get('prod_name', 'No name')
    return ProductInfo(
        article_id=article_id,
        product_name=name,
        image_url=f"/images/0{article_id}.jpg"
    )


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/user/{user_id}/history", response_model=HistoryResponse)
async def get_user_history(user_id: str):
    """Возвращает список купленных пользователем товаров (с картинками)."""
    # Фильтруем транзакции по customer_id
    user_purchases = transactions_df[transactions_df['customer_id'] == user_id]
    if user_purchases.empty:
        return HistoryResponse(user_id=user_id, purchases=[])

    # Берём уникальные товары (можно сортировать по дате последней покупки)
    purchased_items = user_purchases['article_id'].unique()
    products = [get_product_info(int(aid)) for aid in purchased_items]
    return HistoryResponse(user_id=user_id, purchases=products)


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend(user_id: str, top_k: int = 12):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    recs = recommender.recommend(user_id, top_k=top_k)
    products = []
    for article_id, score in recs:
        info = get_product_info(article_id)
        info.score = score
        products.append(info)

    return RecommendResponse(user_id=user_id, recommendations=products)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)