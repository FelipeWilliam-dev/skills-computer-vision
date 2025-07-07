import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from timm import create_model
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Definindo as transformações separadamente
def get_train_transforms(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_data_with_transforms(data_path, img_size, transform):
    """
    Carrega o dataset usando um conjunto de transformações específico.
    """
    dataset = datasets.ImageFolder(data_path, transform=transform)
    num_classes = len(dataset.classes)
    return dataset, num_classes

def load_data2(data_path, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.CIFAR10(root=data_path, train=True, transform=transform)
    num_classes = len(dataset.classes)
    return dataset, num_classes

def define_model(model_name, num_classes):
    model = create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def train_and_validate(model, train_loader, val_loader, device, epochs=10, lr=5e-5, save_path=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs)
    scaler = GradScaler()

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=True)
        for images, labels in train_pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validação]", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast():
                    outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += torch.eq(predicted, labels).sum().item()

        val_acc = 100 * correct / total

        print(f"Epoch [{epoch + 1}/{epochs}] -> Loss: {avg_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                print(f"🎉 Nova melhor acurácia: {best_val_acc:.2f}%. Salvando modelo em '{save_path}'...")
                torch.save(model.state_dict(), save_path)

    print('- - - - - Treinamento finalizado - - - - -')

    return best_val_acc

def generate_report(model, data_loader, device, class_names):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("\n--- Relatório de Classificação ---")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("\n--- Matriz de Confusão ---")
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Verdadeira')
    plt.title('Matriz de Confusão')
    report_path = 'confusion_matrix.png'
    plt.savefig(report_path)
    print(f"Matriz de confusão salva como '{os.path.abspath(report_path)}'")
    plt.show()

def main_percentage_split():
    # --- HIPERPARÂMETROS ---
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 30
    MODEL_NAME = 'convnext_base'
    DATA_PATH = './Dataset3/vehicle/' # Pasta raiz contendo as pastas das classes
    EPOCHS = 15
    LEARNING_RATE = 0.0001
    VALIDATION_SPLIT_SIZE = 0.20 # Porcentagem para o conjunto de validação

    # --- Carregando o dataset completo DUAS VEZES, com transformações diferentes ---
    print('- - - - - Carregando datasets com transformações distintas - - - - -')
    
    # 1. Dataset para extrair índices e classes (pode usar qualquer transformação inicial)
    # Usamos as transformações de validação para esta instância inicial, pois são determinísticas
    base_dataset, num_classes = load_data_with_transforms(DATA_PATH, IMG_SIZE, get_val_transforms(IMG_SIZE))

    # Obtém os índices e os targets (labels) do dataset para a divisão estratificada
    indices = np.arange(len(base_dataset))
    targets = base_dataset.targets 

    # Realiza a divisão estratificada dos ÍNDICES
    train_indices, val_indices = train_test_split(
        indices, test_size=VALIDATION_SPLIT_SIZE, stratify=targets, random_state=42
    )
    
    # Obtém os nomes das classes diretamente do dataset base
    class_names = base_dataset.classes

    # 2. Criar o dataset de TREINO com transformações de aumento de dados
    train_full_dataset, _ = load_data_with_transforms(DATA_PATH, IMG_SIZE, get_train_transforms(IMG_SIZE))
    train_subset = Subset(train_full_dataset, train_indices)

    # 3. Criar o dataset de VALIDAÇÃO com transformações determinísticas (sem aumento de dados)
    val_full_dataset, _ = load_data_with_transforms(DATA_PATH, IMG_SIZE, get_val_transforms(IMG_SIZE))
    val_subset = Subset(val_full_dataset, val_indices)
    
    print(f"Total de amostras no dataset: {len(base_dataset)}")
    print(f"Amostras para Treino (com aumento de dados): {len(train_subset)}")
    print(f"Amostras para Validação (sem aumento de dados): {len(val_subset)}")


    # --- Configurando caminhos e dispositivo ---
    save_dir = 'Save_Models'
    model_save_path = os.path.join(save_dir, f'best_{MODEL_NAME}_split.pth')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Modelos serão salvos em: '{os.path.abspath(model_save_path)}'")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo usado: {device}")

    # --- DataLoaders ---
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

    # --- Treinamento do Modelo ---
    model = define_model(MODEL_NAME, num_classes)

    if int(torch.__version__.split('.')[0]) >= 2:
        model = torch.compile(model)
        print("Modelo compilado com torch.compile() para maior performance.")

    print(f'\n- - - - - Iniciando Treinamento - - - - -')

    best_accuracy = train_and_validate(
        model,
        train_loader,
        val_loader,
        device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        save_path=model_save_path
    )

    print(f"\nA Melhor acurácia de validação alcançada foi: {best_accuracy:.2f}%")

    # --- Geração do Relatório Pós-Treino ---
    print("\n- - - - - Gerando relatório com o melhor modelo salvo - - - - -")

    report_model = define_model(MODEL_NAME, num_classes)
    if int(torch.__version__.split('.')[0]) >= 2:
        report_model = torch.compile(report_model)

    report_model.load_state_dict(torch.load(model_save_path))
    report_model.to(device)

    # Passa os nomes das classes corretos para o relatório
    generate_report(report_model, val_loader, device, class_names)

    print("\nTreinamento concluído.")

if __name__ == '__main__':
    main_percentage_split()