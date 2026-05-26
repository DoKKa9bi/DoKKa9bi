#Тренировка по выбору трёх моделей: RotEye, RotCNN4, RotCNN6

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple
import cv2

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  
DATA_DIR = "data/faces_dataset"                            
LABELS_FILE = "data/labels.csv"                            

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8                                         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4


MODELS_CONFIG = [
    {"name": "RotEyes", "class": None},  
    {"name": "RotCNN4", "class": None},
    {"name": "RotCNN6", "class": None},
]

# =========================================================
# [PARSING] Парсинг данных и подготовка Dataset (np.ndarray -> Tensor)
# =========================================================
def parse_df40_dataset(data_dir: str, train_ratio: float = 0.8) -> Tuple[List[str], List[float], List[str], List[float]]:
    """
    Сканирует DATA_DIR, находит папки 'real' и 'fake', 
    собирает пути к изображениям, назначает метки и делит на train/val.
    """
    paths, labels = [], []
    # Стандартная разметка DF40: real -> 0.0, fake -> 1.0
    label_map = {"real": 0.0, "fake": 1.0}
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for folder_name, label in label_map.items():
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"⚠️  Папка '{folder_path}' не найдена. Пропускаю.")
            continue
            
        # Рекурсивный обход всех изображений внутри папки
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(image_exts):
                    paths.append(os.path.join(root, f))
                    labels.append(label)

    if not paths:
        raise FileNotFoundError(f"В '{data_dir}' не найдено изображений в папках real/ или fake/.")

    # Перемешивание и разделение
    paths = np.array(paths)
    labels = np.array(labels)
    idx = np.random.permutation(len(paths))
    paths, labels = paths[idx], labels[idx]

    split_idx = int(len(paths) * train_ratio)
    return (
        paths[:split_idx].tolist(), labels[:split_idx].tolist(),
        paths[split_idx:].tolist(), labels[split_idx:].tolist()
    )


# =========================================================
# [DATASET] Только чтение в np.ndarray, без преобразований
# =========================================================
class EyesDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[float]):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Сырой np.ndarray (H, W, 3, uint8, BGR)
        img_np = cv2.imread(self.image_paths[idx])
        if img_np is None:
            raise FileNotFoundError(f"Не удалось прочитать: {self.image_paths[idx]}")
            
        # 2. Метка
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Возвращаем сырой массив. DataLoader самостоятельно сконвертирует в Tensor при батчинге.
        return img_np, label


def load_data_split(data_dir: str, labels_file: str, train_ratio: float = 0.8) -> Tuple[List, List, List, List]:
    """Загрузка путей и меток, разделение на train/val"""
    import pandas as pd
    df = pd.read_csv(labels_file, header=None, names=["path", "label"])
    df["full_path"] = df["path"].apply(lambda p: os.path.join(data_dir, p))
    
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(shuffled) * train_ratio)
    
    return (
        shuffled.loc[:split_idx-1, "full_path"].tolist(),
        shuffled.loc[:split_idx-1, "label"].tolist(),
        shuffled.loc[split_idx:, "full_path"].tolist(),
        shuffled.loc[split_idx:, "label"].tolist()
    )


def train_model(model_class, train_loader: DataLoader, val_loader: DataLoader, 
                epochs: int, lr: float, model_name: str) -> Tuple[nn.Module, Dict, float]:
    """Обучает модель, возвращает объект, историю метрик и время обучения."""
    model = model_class().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        # ── TRAIN ──
        model.train()
        train_correct, train_total, train_loss_sum = 0, 0, 0.0
        
        pbar = tqdm(train_loader, desc=f"[{model_name}] Train", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # <-- Вызов модели
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            
            preds = (torch.sigmoid(outputs) > 0.5).float().squeeze(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss_sum += loss.item()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # ── VALIDATION ──
        model.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(1), labels)
                
                preds = (torch.sigmoid(outputs) > 0.5).float().squeeze(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss_sum += loss.item()
        
        # Логирование
        history["train_acc"].append(train_correct / train_total)
        history["val_acc"].append(val_correct / val_total)
        history["train_loss"].append(train_loss_sum / len(train_loader))
        history["val_loss"].append(val_loss_sum / len(val_loader))
        
        print(f"  Epoch {epoch+1:2d} | Train Acc: {history['train_acc'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}")
    
    training_time = time.perf_counter() - start_time
    return model, history, training_time


def measure_inference_time(model: nn.Module, test_loader: DataLoader, n_batches: int = 10) -> Tuple[float, float]:
    """Измеряет время инференса на GPU/CPU."""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= n_batches:
                break
            inputs = inputs.to(DEVICE)
            
            # Прогрев и точный замер
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            start = time.perf_counter()
            _ = model(inputs)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
                
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
            
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"{model.__class__.__name__}: Inference {avg_time:.2f} ± {std_time:.2f} ms")
    return avg_time, std_time


def plot_accuracy_curves(all_histories: Dict[str, Dict], output_path: str = "results/accuracy_curves.png"):
    """График точности по эпохам для всех моделей."""
    plt.figure(figsize=(10, 6))
    for name, hist in all_histories.items():
        epochs = range(1, len(hist["val_acc"]) + 1)
        plt.plot(epochs, hist["val_acc"], label=f"{name} (val)", marker='o', linewidth=2)
        plt.plot(epochs, hist["train_acc"], label=f"{name} (train)", linestyle='--', alpha=0.7)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Validation & Train Accuracy over Epochs", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_table(all_metrics: Dict[str, Dict], output_path: str = "results/summary_metrics.png"):
    """Сводный график времени обучения и инференса."""
    names = list(all_metrics.keys())
    train_times = [m["training_time"] for m in all_metrics.values()]
    infer_times = [m["inference_time"] for m in all_metrics.values()]
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_times, width, label='Train Time (s)', color='#4C72B0', alpha=0.8)
    ax1.set_ylabel('Train Time (seconds)')
    ax1.tick_params(axis='y')
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, infer_times, width, label='Inference Time (ms)', color='#DD8452', alpha=0.8)
    ax2.set_ylabel('Inference Time (ms)')
    ax2.tick_params(axis='y')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11)
    ax1.set_title("Training vs Inference Time Comparison", fontsize=14)
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.grid(True, alpha=0.3, axis='y')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # 1. Загрузка данных
    train_paths, train_labels, val_paths, val_labels = load_data_split(DATA_DIR, LABELS_FILE, TRAIN_SPLIT)
    print(f"Dataset: {len(train_paths)} train, {len(val_paths)} val images")
    
    train_loader = DataLoader(EyesDataset(train_paths, train_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(EyesDataset(val_paths, val_labels),   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    all_histories = {}
    all_metrics   = {}
    
    # 2. Последовательное обучение и оценка
    for config in MODELS_CONFIG:
        model_name = config["name"]
        model_cls  = config["class"]
        
        if model_cls is None:
            print(f"Класс {model_name} не импортирован. Пропускаю.")
            continue
            
        print(f"\n{'='*40}\nStart: {model_name}\n{'='*40}")
        model, history, train_time = train_model(model_cls, train_loader, val_loader, EPOCHS, LEARNING_RATE, model_name)
        inf_time, inf_std = measure_inference_time(model, val_loader)
        
        # Сохранение результатов
        all_histories[model_name] = history
        all_metrics[model_name] = {
            "best_val_acc": max(history["val_acc"]),
            "training_time": train_time,
            "inference_time": inf_time,
            "inference_std": inf_std
        }
        
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/{model_name}_final.pt")
        print(f"Модель сохранена: models/{model_name}_final.pt")

    # 3. Визуализация
    print("\nГенерация графиков...")
    plot_accuracy_curves(all_histories)
    plot_summary_table(all_metrics)
    
    # 4. Итоговая сводка
    print("\n" + "="*50)
    print("PROFIT?")
    print("="*50)
    print(f"{'Модель':<12} | {'Val Acc':<8} | {'Train Time':<10} | {'Inference Time'}")
    print("-"*50)
    for name, m in all_metrics.items():
        print(f"{name:<12} | {m['best_val_acc']:.4f}     | {m['training_time']:.1f}s      | {m['inference_time']:.2f}±{m['inference_std']:.2f} ms")
    print("="*50)

if __name__ == "__main__":
    
    from model import see_eyes, RotEyes, RotCNN4, RotCNN6    
    main()
