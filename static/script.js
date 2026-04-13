document.getElementById('loadBtn').addEventListener('click', async () => {
    const userId = document.getElementById('userId').value.trim();
    if (!userId) {
        alert('Введите ID пользователя');
        return;
    }

    // Показать загрузку
    const historyGrid = document.getElementById('historyGrid');
    const recGrid = document.getElementById('recGrid');
    historyGrid.innerHTML = '<div class="loading">Загрузка истории...</div>';
    recGrid.innerHTML = '<div class="loading">Загрузка рекомендаций...</div>';

    // Загружаем историю и рекомендации параллельно
    try {
        const [historyRes, recRes] = await Promise.all([
            fetch(`/user/${userId}/history`),
            fetch(`/recommend/${userId}?top_k=12`)
        ]);

        if (!historyRes.ok) throw new Error(`History error: ${historyRes.status}`);
        if (!recRes.ok) throw new Error(`Rec error: ${recRes.status}`);

        const historyData = await historyRes.json();
        const recData = await recRes.json();

        // Отрисовка истории
        renderProducts(historyGrid, historyData.purchases, false);
        // Отрисовка рекомендаций (с оценками)
        renderProducts(recGrid, recData.recommendations, true);
    } catch (err) {
        console.error(err);
        historyGrid.innerHTML = `<div class="error">Ошибка: ${err.message}</div>`;
        recGrid.innerHTML = `<div class="error">Ошибка: ${err.message}</div>`;
    }
});

function renderProducts(container, products, showScore) {
    if (!products || products.length === 0) {
        container.innerHTML = '<div class="empty">Товары не найдены</div>';
        return;
    }

    const grid = document.createElement('div');
    grid.className = 'grid';

    products.forEach(prod => {
        const card = document.createElement('div');
        card.className = 'card';

        // Изображение с fallback
        const img = document.createElement('img');
        img.src = prod.image_url;
        img.onerror = () => { img.src = '/static/placeholder.png'; img.onerror = null; };
        img.alt = prod.product_name;

        const nameSpan = document.createElement('div');
        nameSpan.className = 'product-name';
        nameSpan.textContent = prod.product_name.length > 40 ? prod.product_name.slice(0,37)+'...' : prod.product_name;

        const idSpan = document.createElement('div');
        idSpan.className = 'article-id';
        idSpan.textContent = `Арт. ${prod.article_id}`;

        card.appendChild(img);
        card.appendChild(nameSpan);
        card.appendChild(idSpan);

        if (showScore && prod.score !== undefined) {
            const scoreSpan = document.createElement('div');
            scoreSpan.className = 'score';
            scoreSpan.textContent = `релевантность: ${prod.score.toFixed(2)}`;
            card.appendChild(scoreSpan);
        }

        grid.appendChild(card);
    });

    container.innerHTML = '';
    container.appendChild(grid);
}