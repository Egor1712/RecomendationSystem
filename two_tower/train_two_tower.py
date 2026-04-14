import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

import consts
from src.dataset_preprocess import build_dataset_with_matrix
from src.utils import load_data
from two_tower.model import HMDataset, TwoTowerModel, train_two_tower_epoch, map_at_k_two_tower


class CFG:
    data_output = './output'
    experiment_name = '1'
    lightfm_params = {
        'loss': 'warp',
        "learning_rate": 0.05,
        'no_components': 256,
        'user_alpha': 1e-6,
        'item_alpha': 1e-6,
        'random_state': 42
    }

os.makedirs(CFG.data_output, exist_ok=True)
os.makedirs(os.path.join(CFG.data_output, CFG.experiment_name), exist_ok=True)

def train_two_tower_with_logging(interaction,
                                 epochs=30,
                                 batch_size=1024, embedding_dim=128,
                                 num_negatives=5,
                                 learning_rate=0.001,
                                 validation_size=0.2,
                                 k=12,
                                 save_best=True):
    experiment_path = os.path.join(CFG.data_output, CFG.experiment_name)
    # Разделение на train/validation
    train_interaction, val_interaction = train_test_split(interaction, test_size=validation_size, random_state=42)
    train_dataset = HMDataset(train_interaction, num_negatives=num_negatives)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    model = TwoTowerModel(interaction.shape[0], interaction.shape[1], embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log_file = os.path.join(experiment_path, 'training_log.csv')
    history = {'epoch': [], 'train_loss': [], 'map@12': []}
    best_map = -1.0

    epoch_iterator = tqdm(range(1, epochs + 1), desc="Обучение Two-Tower", unit="epoch")
    for epoch in epoch_iterator:
        loss = train_two_tower_epoch(model, train_loader, optimizer, device)
        map_score = map_at_k_two_tower(model, val_interaction, k=k, batch_size=batch_size, device=device)
        history['epoch'].append(epoch)
        history['train_loss'].append(loss)
        history['map@12'].append(map_score)
        epoch_iterator.set_postfix({"Loss": f"{loss:.4f}", "MAP@12": f"{map_score:.6f}"})
        if save_best and map_score > best_map:
            best_map = map_score
            torch.save(model.state_dict(), os.path.join(experiment_path,  'two_tower_best.pt'))
            tqdm.write(f"  -> Новая лучшая модель сохранена (MAP@{k}={map_score:.6f})")

    pd.DataFrame(history).to_csv(log_file, index=False)
    print(f"Лог сохранён в {log_file}")
    # График MAP@12
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['map@12'], marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel(f'MAP@{k}')
    plt.title(f'Two-Tower: MAP@{k} on Validation')
    plt.grid(True)
    plt.savefig(os.path.join(experiment_path, 'map_curve.png'), dpi=150)
    plt.show()
    # График loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], marker='x', linestyle='-', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss (BPR)')
    plt.title('Two-Tower: Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(experiment_path, 'loss_curve.png'), dpi=150)
    plt.show()
    return model, history


if __name__ == "__main__":
    print("=== Загрузка данных ===")
    transactions, articles, customers, sample_sub = load_data(consts.WORKING_DATASET_DIRECTORY, CFG.data_output)
    print("=== Подготовка признаков ===")
    interaction, user_features, item_features, _, _ = build_dataset_with_matrix(articles, customers, transactions, CFG.data_output)
    print("=== Обучение модели ===")
    train_two_tower_with_logging(interaction)


