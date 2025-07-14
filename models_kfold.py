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
from sklearn.metrics import roc_curve, auc
from itertools import cycle

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
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validação]", leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast():
                    outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += torch.eq(predicted, labels).sum().item()

        val_acc = 100 * correct / total

        tqdm.write(f"Epoch [{epoch + 1}/{epochs}] -> Loss: {avg_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                # CORREÇÃO: Usando tqdm.write aqui também
                tqdm.write(f"🎉 Nova melhor acurácia: {best_val_acc:.2f}%. Salvando modelo em '{save_path}'...")
                torch.save(model.state_dict(), save_path)

    # Este print está fora do loop principal de épocas, então pode ser um print normal.
    print('\n- - - - - Treinamento finalizado - - - - -')

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

def generate_roc_curves(model, data_loader, device, num_classes, class_names):
    """Gera e plota as curvas ROC para cada classe."""
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            scores = torch.nn.functional.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_scores.extend(scores.cpu().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Binariza os rótulos para o cálculo da ROC multiclasse
    y_true_bin = nn.functional.one_hot(torch.from_numpy(y_true), num_classes=num_classes).numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotar as curvas ROC
    plt.figure(figsize=(12, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    print(f"Curvas ROC salvas como '{os.path.abspath('roc_curves.png')}'")
    plt.show()

def run_stratified_kfold(dev_indices, all_targets, DATA_PATH, IMG_SIZE, BATCH_SIZE, MODEL_NAME, EPOCHS, LEARNING_RATE,
                         N_SPLITS):
    """
    Executa a validação cruzada Stratified K-Fold em um conjunto específico de índices (o conjunto de desenvolvimento).
    """
    print("\n--- PASSO 2: Executando Validação Cruzada no Conjunto de Desenvolvimento ---")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_accuracies = []

    dev_targets = all_targets[dev_indices]

    save_dir = 'Save_Models_KFold'
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo para o K-Fold: {device}")

    for fold, (train_pos_indices, val_pos_indices) in enumerate(skf.split(dev_indices, dev_targets)):
        print(f"\n- - - - - [ K-FOLD {fold + 1}/{N_SPLITS} ] - - - - -")

        train_indices = dev_indices[train_pos_indices]
        val_indices = dev_indices[val_pos_indices]

        train_full_dataset, _ = load_data_with_transforms(DATA_PATH, IMG_SIZE, get_train_transforms(IMG_SIZE))
        train_subset = Subset(train_full_dataset, train_indices)

        val_full_dataset, num_classes = load_data_with_transforms(DATA_PATH, IMG_SIZE, get_val_transforms(IMG_SIZE))
        val_subset = Subset(val_full_dataset, val_indices)

        print(f"Amostras de Treino do Fold: {len(train_subset)}, Amostras de Validação do Fold: {len(val_subset)}")

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

        model = define_model(MODEL_NAME, num_classes)
        if int(torch.__version__.split('.')[0]) >= 2:
            model = torch.compile(model)

        model_save_path = os.path.join(save_dir, f'best_model_fold_{fold + 1}.pth')
        best_fold_acc = train_and_validate(
            model, train_loader, val_loader, device, EPOCHS, LEARNING_RATE, model_save_path
        )
        fold_accuracies.append(best_fold_acc)
        print(f"Melhor acurácia para o Fold {fold + 1}: {best_fold_acc:.2f}%")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print(f"\n- - - - - Resultados Finais da Validação Cruzada ({N_SPLITS} folds) - - - - -")
    print(f"Acurácias de cada fold: {[f'{acc:.2f}%' for acc in fold_accuracies]}")
    print(f"Acurácia Média Estimada: {mean_acc:.2f}%")
    print(f"Desvio Padrão: {std_acc:.4f}")
    print("--- Fim do Passo 2 ---")

def main_production_workflow():
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    MODEL_NAME = 'convnext_base'
    DATA_PATH = './Dataset_tratado2/vehicle/'
    EPOCHS = 15
    LEARNING_RATE = 0.0001
    TEST_SPLIT_SIZE = 0.15
    N_SPLITS_KFold = 5

    # --- PASSO 1: DIVISÃO INICIAL EM DESENVOLVIMENTO (TRAIN+VAL) E TESTE ---
    print("--- PASSO 1: Dividindo o dataset em Desenvolvimento e Teste ---")
    base_dataset, num_classes = load_data_with_transforms(DATA_PATH, IMG_SIZE, get_val_transforms(IMG_SIZE))
    indices = np.arange(len(base_dataset))
    targets = np.array(base_dataset.targets)
    class_names = base_dataset.classes

    dev_indices, test_indices = train_test_split(
        indices, test_size=TEST_SPLIT_SIZE, stratify=targets, random_state=42
    )
    print(f"Total de amostras: {len(base_dataset)}")
    print(f"Amostras para Desenvolvimento: {len(dev_indices)}")
    print(f"Amostras para Teste Final: {len(test_indices)}")

    # --- PASSO 2: RODAR K-FOLD NO CONJUNTO DE DESENVOLVIMENTO ---
    run_stratified_kfold(
        dev_indices=dev_indices,
        all_targets=targets,
        DATA_PATH=DATA_PATH,
        IMG_SIZE=IMG_SIZE,
        BATCH_SIZE=BATCH_SIZE,
        MODEL_NAME=MODEL_NAME,
        EPOCHS=EPOCHS,
        LEARNING_RATE=LEARNING_RATE,
        N_SPLITS=N_SPLITS_KFold
    )

    # --- PASSO 3: TREINAR O MODELO FINAL NO CONJUNTO DE DESENVOLVIMENTO ---
    print("\n--- PASSO 3: Treinando o modelo final em TODO o conjunto de Desenvolvimento ---")

    # Para o treino final, usamos todos os dev_indices para treinar e uma pequena parte para validar
    train_final_indices, val_final_indices = train_test_split(
        dev_indices, test_size=0.1, stratify=targets[dev_indices], random_state=42
    )

    train_full_dataset, _ = load_data_with_transforms(DATA_PATH, IMG_SIZE, get_train_transforms(IMG_SIZE))
    final_train_subset = Subset(train_full_dataset, train_final_indices)

    val_full_dataset, _ = load_data_with_transforms(DATA_PATH, IMG_SIZE, get_val_transforms(IMG_SIZE))
    final_val_subset = Subset(val_full_dataset, val_final_indices)

    final_train_loader = DataLoader(final_train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    final_val_loader = DataLoader(final_val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    save_dir = 'Final_Model'
    model_save_path = os.path.join(save_dir, f'production_{MODEL_NAME}.pth')
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = define_model(MODEL_NAME, num_classes)
    if int(torch.__version__.split('.')[0]) >= 2:
        model = torch.compile(model)

    train_and_validate(model, final_train_loader, final_val_loader, device, EPOCHS, LEARNING_RATE, model_save_path)

    # --- PASSO 4: O EXAME FINAL - AVALIAR NO CONJUNTO DE TESTE ---
    print("\n--- PASSO 4: Avaliação final no conjunto de Teste ---")

    # Criamos o DataLoader de teste com o subconjunto que guardamos no início
    test_subset = Subset(val_full_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    final_model = define_model(MODEL_NAME, num_classes)
    if int(torch.__version__.split('.')[0]) >= 2:
        final_model = torch.compile(final_model)

    final_model.load_state_dict(torch.load(model_save_path))
    final_model.to(device)

    generate_report(final_model, test_loader, device, class_names)
    generate_roc_curves(final_model, test_loader, device, num_classes, class_names)

    print("\nProcesso de produção concluído.")


if __name__ == '__main__':
    #main_percentage_split()
    #main_stratified_kfold()
    main_production_workflow()