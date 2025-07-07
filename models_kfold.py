import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from timm import create_model
from sklearn.model_selection import StratifiedKFold, train_test_split #<- importação atualizada
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def load_data(data_path, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Carregar o dataset completo
    dataset = datasets.ImageFolder(data_path, transform=transform)
    num_classes = len(dataset.classes)

    return dataset, num_classes

def load_data2(data_path, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Carregar o dataset completo
    dataset = datasets.CIFAR10(root=data_path, train=True, transform=transform) #<- treinamento definido
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
            # CORREÇÃO APLICADA AQUI
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
                # CORREÇÃO APLICADA AQUI TAMBÉM
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
    # ... (cole a função generate_report da nossa conversa anterior aqui) ...
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

def main():
    models_list = ['beit_base_patch16_224',
                   'convformer_s18',
                   'convnext_base',
                   'deit_base_patch16_224',
                   'efficientformer_l1',
                   'maxvit_tiny_tf_224',
                   'mixer_b16_224','mobilevit_xs',
                   'swinv2_base_window12_192_22k',
                   'vit_base_patch16_224']

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    MODEL_NAME = 'convnext_base'
    DATA_PATH = './Dataset3/vehicle/'
    EPOCHS = 15
    LEARNING_RATE = 0.0001  # 5e-5
    N_SPLITS = 5
    #SUBSET_SIZE = 1500

    print('- - - - - Carregando os  dados - - - - - -')
    #full_dataset, num_classes = load_data2(DATA_PATH, IMG_SIZE)
    dataset, num_classes = load_data(DATA_PATH, IMG_SIZE)

    targets = dataset.targets

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


    # --- INÍCIO DO BLOCO DE ALTERAÇÃO ---
    #print(f'Criando um subconjunto estratificado de {SUBSET_SIZE} imagens.')
    # Gerar índices para realizar a divisão estratificada
    #indices = list(range(len(full_dataset)))
    #targets = full_dataset.targets

    # Usar train_test_split para obter um subconjunto estratificado de índices
    # O segundo valor de retorno (com '_') é descartado, pois não precisamos dele
    """subset_indices, _ = train_test_split(
        indices,
        train_size=SUBSET_SIZE,  # Define o tamanho absoluto do subset
        stratify=targets,
        random_state=42  # Para resultados reprodutíveis
    )

    dataset = Subset(full_dataset, subset_indices)

    # Precisamos dos alvos (targets) correspondentes ao nosso novo subconjunto para o StratifiedKFold
    subset_targets = [targets[i] for i in subset_indices]

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)"""


    print('- - - - - - - - - - - - - - - - - - - - -')
    print(f"Número de classes: {num_classes}")
    print(f"Quantidade total de imagens no dataset: {len(dataset)}")
    print('- - - - - - - - - - - - - - - - - - - - -')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo usado: {device}")

    accuracies = []

    print(f'\n- - - - - Iniciando Treinamento com Validação Cruzada Usando o modelo: {MODEL_NAME} - - - - -')

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)), targets)):
        print(f"\n- - - - Fold {fold + 1}/{N_SPLITS} - - - -")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        num_train_files = len(train_subset)
        num_val_files = len(val_subset)
        print(f"Imagens para Treinamento neste fold: {num_train_files}")
        print(f"Imagens para Validação neste fold: {num_val_files}")

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

        model = define_model(MODEL_NAME, num_classes)
        print(f"Treinando fold {fold + 1} por {EPOCHS} épocas...")

        val_acc = train_and_validate(
            model,
            train_loader,
            val_loader,
            device,
            epochs=EPOCHS,
            lr=LEARNING_RATE
        )
        accuracies.append(val_acc)
        print(f"Acurácia Média - fold[{fold + 1}]: {val_acc:.2f}%")

    print(f"\n- - - - Resultados Finais da Validação Cruzada ({N_SPLITS} folds) - - - -")
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print(f"- - - - Resultados da Validação Cruzada - - - -")
    print(f"Modelo treinado: {MODEL_NAME}")
    print(f"Acurácia Média: {mean_acc:.2f}%")
    print(f"Desvio Padrão da Acurácia: {std_acc:.4f}%")

    # --- Salvando o relatório final em um arquivo .txt ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f'final_report_{MODEL_NAME}_{timestamp}.txt'

    hyperparameters = {
        'Tamanho da Imagem': IMG_SIZE,
        'Tamanho do Batch': BATCH_SIZE,
        'Épocas': EPOCHS,
        'Taxa de Aprendizagem': LEARNING_RATE,
        'Número de Folds (Splits)': N_SPLITS,
        'Dispositivo': str(device)
    }

    with open(results_filename, 'w', encoding='utf-8') as f:
        f.write(f"--- Relatório do Treino ({timestamp}) ---\n")
        f.write(f"Modelo: {MODEL_NAME}\n\n")

        f.write("== Resultados Finais ==\n")
        f.write(f"Acurácia Média Final: {mean_acc:.2f}%\n")
        f.write(f"Desvio Padrão da Acurácia: {std_acc:.4f}\n\n")

        f.write("== Hiperparâmetros ==\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")

    print(f"\nRelatório final salvo em: {results_filename}")


def main_percentage_split():
    # --- HIPERPARÂMETROS ---
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 30
    MODEL_NAME = 'convnext_base'
    DATA_PATH = './Dataset3/vehicle/'
    EPOCHS = 15
    LEARNING_RATE = 0.0001
    VALIDATION_SPLIT_SIZE = 0.20

    # --- Carregando os dados e criando a divisão ---
    print('- - - - - Carregando os dados e criando a divisão - - - - -')
    dataset, num_classes = load_data(DATA_PATH, IMG_SIZE)
    indices = np.arange(len(dataset))
    targets = dataset.targets
    train_indices, val_indices = train_test_split(
        indices, test_size=VALIDATION_SPLIT_SIZE, stratify=targets, random_state=42
    )
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    print(f"Total: {len(dataset)}, Treino: {len(train_subset)}, Validação: {len(val_subset)}")

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
    # CORREÇÃO: Definimos apenas o modelo que será treinado
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

    # CORREÇÃO: Criamos o modelo de relatório aqui e o compilamos
    report_model = define_model(MODEL_NAME, num_classes)
    if int(torch.__version__.split('.')[0]) >= 2:
        report_model = torch.compile(report_model)  # Compila o modelo antes de carregar!

    # Agora o carregamento funcionará
    report_model.load_state_dict(torch.load(model_save_path))
    report_model.to(device)

    # Gera o relatório de classificação e a matriz de confusão
    generate_report(report_model, val_loader, device, dataset.classes)

    print("\nTreinamento concluído.")


if __name__ == "__main__":
    main_percentage_split()

