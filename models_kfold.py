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


def train_and_validate(model, train_loader, val_loader, device, epochs=10, lr=5e-5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=True )
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validação]", leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += torch.eq(predicted, labels).sum().item()

        val_acc = 100 * correct / total

        print(f"Epoch [{epoch + 1}/{epochs}] -> Loss: {avg_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

        accuracies.append(val_acc)

    print('- - - - - Treinamento finalizado - - - - -')

    return np.mean(accuracies)  # retorna a acurácia média de cada época


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

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

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
    BATCH_SIZE = 30  # Mantendo o batch size maior para performance
    MODEL_NAME = 'convnext_base'
    DATA_PATH = './Dataset3/vehicle/'
    EPOCHS = 15
    LEARNING_RATE = 0.0001

    # ALTERAÇÃO: Definimos o tamanho da divisão de validação (20%)
    VALIDATION_SPLIT_SIZE = 0.20

    print('- - - - - Carregando os dados - - - - - -')
    dataset, num_classes = load_data(DATA_PATH, IMG_SIZE)

    print('- - - - - Criando a divisão treino/validação - - - - -')

    # Pegamos os índices e os alvos (labels) para a divisão estratificada
    indices = np.arange(len(dataset))
    targets = dataset.targets

    # ALTERAÇÃO: Usamos train_test_split para criar uma única divisão
    train_indices, val_indices = train_test_split(
        indices,
        test_size=VALIDATION_SPLIT_SIZE,  # Usa o percentual que definimos
        stratify=targets,  # Garante que a proporção de classes seja a mesma nos dois conjuntos
        random_state=42  # Para resultados reprodutíveis
    )

    # Criamos os Subsets a partir dos índices gerados
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    print('- - - - - - - - - - - - - - - - - - - - -')
    print(f"Número de classes: {num_classes}")
    print(f"Total de imagens: {len(dataset)}")
    print(f"Imagens de Treino: {len(train_subset)}")
    print(f"Imagens de Validação: {len(val_subset)}")
    print('- - - - - - - - - - - - - - - - - - - - -')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo usado: {device}")

    # Criamos os DataLoaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

    # Definimos o modelo
    model = define_model(MODEL_NAME, num_classes)
    if int(torch.__version__.split('.')[0]) >= 2:
        model = torch.compile(model)
        print("Modelo compilado com torch.compile() para maior performance.")

    print(
        f'\n- - - - - Iniciando Treinamento com Percentage Split ({100 - VALIDATION_SPLIT_SIZE * 100}/{VALIDATION_SPLIT_SIZE * 100}) - - - - -')

    # ALTERAÇÃO: Chamamos a função de treino apenas uma vez, sem loop de fold
    final_avg_accuracy = train_and_validate(
        model,
        train_loader,
        val_loader,
        device,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    print(f"\n- - - - Resultados Finais do Percentage Split - - - -")
    print(f"Modelo treinado: {MODEL_NAME}")
    print(f"Acurácia Média das épocas no conjunto de validação: {final_avg_accuracy:.2f}%")

    # Você pode querer salvar um relatório diferente ou adaptar o atual
    print("\nTreinamento concluído.")


if __name__ == "__main__":
    main_percentage_split()

