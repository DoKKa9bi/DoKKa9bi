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
from typing import List, Tuple, Dict

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
    {
        "name": "RotEyes",
        "class": RotEyes,           
        "input_mode": "all_5",     
    },
    {
        "name": "RotCNN4",
        "class": RotCNN4,
        "input_mode": "first_only", 
    },
    {
        "name": "RotCNN6",
        "class": RotCNN6,
        "input_mode": "first_only",
    },
]


class EyesDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], predictor_path: str):
        """
       CHECK THIS PART
        """
        self.image_paths = image_paths
        self.labels = labels
        self.predictor_path = predictor_path
        self.cache = {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import cv2
        img_bgr = cv2.imread(self.image_paths[idx])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        tensors = see_eyes(img_rgb, self.predictor_path) 
  
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return tensors, label


def load_data_split(data_dir: str, labels_file: str, train_ratio: float = 0.8):
    """
    CSV parse for dataset
    """
    import pandas as pd
    df = pd.read_csv(labels_file, header=None, names=["path", "label"])
    
    # Полные пути
    df["full_path"] = df["path"].apply(lambda p: os.path.join(data_dir, p))
    
    # Разделение
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(shuffled) * train_ratio)
    
    train_paths = shuffled.loc[:split_idx-1, "full_path"].tolist()
    train_labels = shuffled.loc[:split_idx-1, "label"].tolist()
    val_paths = shuffled.loc[split_idx:, "full_path"].tolist()
    val_labels = shuffled.loc[split_idx:, "label"].tolist()
    
    return train_paths, train_labels, val_paths, val_labels
#?????????????????????????????????????????????????????????


def train_model(model_config: dict, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float):
    """
    Обучает одну модель, возвращает историю метрик и время обучения.
    """
    ModelClass = model_config["class"]
    input_mode = model_config["input_mode"]
    model_name = model_config["name"]
    
    
    model = ModelClass().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # ── TRAIN ──
        model.train()
        train_correct, train_total, train_loss_sum = 0, 0, 0.0
        
        for tensors_batch, labels_batch in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs} [TRAIN]"):
            labels_batch = labels_batch.to(DEVICE)
            
            if input_mode == "all_5":
                inputs = [t.to(DEVICE) for t in tensors_batch] 
                outputs = model(*inputs) 
            else:  
                input_1 = tensors_batch[0].to(DEVICE)  
                outputs = model(input_1)
            
            loss = criterion(outputs.squeeze(1), labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds.squeeze(1) == labels_batch).sum().item()
            train_total += labels_batch.size(0)
            train_loss_sum += loss.item()
        
        # ── VALIDATION ──
        model.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        
        with torch.no_grad():
            for tensors_batch, labels_batch in val_loader:
                labels_batch = labels_batch.to(DEVICE)
                
                if input_mode == "all_5":
                    inputs = [t.to(DEVICE) for t in tensors_batch]
                    outputs = model(*inputs)
                else:
                    input_1 = tensors_batch[0].to(DEVICE)
                    outputs = model(input_1)
                
                loss = criterion(outputs.squeeze(), labels_batch)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds.squeeze(1) == labels_batch).sum().item()
                val_total += labels_batch.size(0)
                val_loss_sum += loss.item()
        
        # Сохранение метрик
        history["train_acc"].append(train_correct / train_total)
        history["val_acc"].append(val_correct / val_total)
        history["train_loss"].append(train_loss_sum / len(train_loader))
        history["val_loss"].append(val_loss_sum / len(val_loader))
        
        print(f"  [{model_name}] Epoch {epoch+1}: "
              f"train_acc={history['train_acc'][-1]:.4f}, "
              f"val_acc={history['val_acc'][-1]:.4f}")
    
    training_time = time.time() - start_time
    print(f"{model_name}--{training_time:.2f}s")
    
    return model, history, training_time


def measure_inference_time(model, model_config: dict, test_loader: DataLoader, n_batches: int = 10):
    """
    CHECK
    """
    model.eval()
    input_mode = model_config["input_mode"]
    times = []
    
    with torch.no_grad():
        for i, (tensors_batch, _) in enumerate(test_loader):
            if i >= n_batches:
                break
                
            if input_mode == "all_5":
                inputs = [t.to(DEVICE) for t in tensors_batch]
            else:
                inputs = [tensors_batch[0].to(DEVICE)]
            
            
            if i == 0:
                _ = model(*inputs) if input_mode == "all_5" else model(inputs[0])
            
            
            start = time.time()
            _ = model(*inputs) if input_mode == "all_5" else model(inputs[0])
            torch.cuda.synchronize() if DEVICE.type == "cuda" else None
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"{model_config['name']}: time is {avg_time:.2f} ± {std_time:.2f} ms per image")
    return avg_time, std_time



def plot_accuracy_curves(all_histories: Dict[str, dict], output_path: str = "results/accuracy_curves.png"):
    """Graph"""
    plt.figure(figsize=(12, 6))
    for name, hist in all_histories.items():
        epochs = range(1, len(hist["val_acc"]) + 1)
        plt.plot(epochs, hist["val_acc"], label=f"{name} (val)", marker='o')
        plt.plot(epochs, hist["train_acc"], label=f"{name} (train)", linestyle='--', alpha=0.7)
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("test")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"table: {output_path}")


def plot_summary_table(all_metrics: Dict[str, dict], output_path: str = "results/summary_table.png"):
    """Table"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    rows = [[
        metrics["best_val_acc"],
        metrics["training_time"],
        metrics["inference_time"]
    ] for metrics in all_metrics.values()]
    
    columns = ["Accuracy", "Train time", "use time"]
    table = ax.table(cellText=rows,
                     rowLabels=list(all_metrics.keys()),
                     colLabels=columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"table: {output_path}")


def main():
    
    train_paths, train_labels, val_paths, val_labels = load_data_split(
        DATA_DIR, LABELS_FILE, TRAIN_SPLIT
    )
    
    train_dataset = EyesDataset(train_paths, train_labels, PREDICTOR_PATH)
    val_dataset = EyesDataset(val_paths, val_labels, PREDICTOR_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    all_histories = {}
    all_metrics = {}
    
    for config in MODELS_CONFIG:
        print(f"\nTraining: {config['name']}")
        model, history, train_time = train_model(
            config, train_loader, val_loader, EPOCHS, LEARNING_RATE
        )
        
        
        inf_time, inf_std = measure_inference_time(model, config, val_loader)
        
        
        all_histories[config["name"]] = history
        all_metrics[config["name"]] = {
            "best_val_acc": max(history["val_acc"]),
            "training_time": train_time,
            "inference_time": inf_time,
            "inference_std": inf_std,
        }
        
        
        torch.save(model.state_dict(), f"models/{config['name']}_final.pt")
    
    
    print("\nGenerating...")
    plot_accuracy_curves(all_histories)
    plot_summary_table(all_metrics)
    
    
    print("\n"+"_"=*50)
    print("PROFIT?")
    print(" _"*50)
    for name, m in all_metrics.items():
        print(f"{name:12} | val_acc: {m['best_val_acc']:.4f} | "
              f"train: {m['training_time']:.1f}s | infer: {m['inference_time']:.2f}±{m['inference_std']:.2f}ms")
    print("_"*50)


if __name__ == "__main__":
    
    from model import see_eyes, RotEyes, RotCNN4, RotCNN6    
    main()
