# 가중치 0.00005로 낮추고 
# 모든 층을 학습 불가능하게 freeze 한 다음 
# FC레이어만 교체
# 학습률 스케줄링 적용 (epoch 3까지 웜업)
# 데이터 증강 추가
# 학습률 감쇠 추가

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import datetime
import sys
import matplotlib.pyplot as plt
import random
import numpy as np

# 학습 후 가중치 파일 저장을 위해 시용함
now = datetime.datetime.now().strftime('%Y%m%d_%H%M')

if __name__ == '__main__':
    # GPU 사용 설정
    print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"연결된 그래픽카드: {torch.cuda.get_device_name(0)}")
    else:
        print("현재 CPU만 사용 중입니다.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터셋 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_dir = os.path.join(current_dir, 'dataset', 'train')
    val_dir = os.path.join(current_dir, 'dataset', 'val')

    save_path = os.path.join(current_dir, f'best_resnet50_{now}.pth')
    
    print(f"Train 경로: {train_dir}")
    print(f"Val 경로: {val_dir}")
    print("-"*50)

    # 2. 하이퍼파라미터 및 전처리 설정
    BATCH_SIZE = 16  # 메모리 부족하면 줄이기
    LEARNING_RATE = 0.00005
    NUM_EPOCHS = 20
    WARMUP_EPOCHS = 3 # 초반 epoch 3동안 웜업 진행


    # ImageNet 학습 때 사용된 정규화 값(표준) 그대로 적용
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 3. 데이터 로더 생성 (이미지 불러오기)
    # ImageNet 폴더 구조: [Root] -> [Class] -> [Image]
    try:
        image_datasets = {
            'train': datasets.ImageFolder(train_dir, data_transforms['train']),
            'val': datasets.ImageFolder(val_dir, data_transforms['val'])
        }
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
            'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
    
        print("데이터 로드 완료")
        print(f"  -분류할 클래스: {class_names}")
        print(f"  -학습 데이터: {dataset_sizes['train']}장")
        print(f"  -검증 데이터: {dataset_sizes['val']}장")
        print(f"Peak Learning Rate: {LEARNING_RATE}")
        print(f"Warmup Epochs: {WARMUP_EPOCHS}")

        print("-"*20, "파라미터", "-"*20)
        print(f"   -batch size: {BATCH_SIZE}")
        print(f"   -Learning Rate:{LEARNING_RATE}")
        print(f"   -Epoch: {NUM_EPOCHS}")
        print("-"*50)
    
    except Exception as e:
        print("데이터 로드 에러")
        import sys
        sys.exit()
    
    # 4. ResNet50 모델 불러오기 및 수정
    # ImageNet 가중치(pretrained) 사용
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # 기존 학습된 feature extractor freeze
    for param in model.parameters():
        param.requires_grad = False

    # layer 4 풀기
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # 마지막 FC layer를 PlantVillage 데이터 개수에 맞게 교체
    num_frts = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_frts, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(class_names)) # 클래스 개수 자동 인식
    )
    model = model.to(device)

    # 5. Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)

    # 스케줄러 정의
    # 웜업
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)

    # decay
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

    # 두 스케줄러를 순차적으로 연결
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[WARMUP_EPOCHS])

    print(f"학습 시작 (총 {NUM_EPOCHS} epoch)")

    # 기록 저장을 위한 리스트
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # 최고 정확도 기록용
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        # 현재 학습률 출력(확인용)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'epoch {epoch + 1}/{NUM_EPOCHS} | LR: {current_lr:.6f}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'    -[{phase}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            # 기록 저장
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # val 성능이 신기록이면 저장 
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 모델의 가중치를 파일로 저장
                torch.save(model.state_dict(), save_path)
        scheduler.step()

    print(" 모든 학습이 완료")
    print(f"최종 저장된 파일: {save_path}")

    # 결과 그래프 시각화
    plt.figure(figsize=(10, 5))
    
    # Accuracy 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    # Loss 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig(os.path.join(current_dir, f'result_graph_{now}.png'))
    print(f"결과 그래프 저장됨: result_graph_{now}.png")
