import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys


# 1. train.py와 동일하게 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습할 때 생성된 클래스 이름 리스트
def load_class_names(train_dir):
    try:
        # 폴더 이름들을 알파벳순으로 정렬 
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        
        if not classes:
            print("오류: train 폴더에 클래스 폴더가 없습니다.")
            sys.exit()
            
        print(f"클래스 목록 로드 완료 ({len(classes)}개)")
        print(f"   - 0번(첫번째): {classes[0]}")
        print(f"   - 마지막: {classes[-1]}")
        return classes
        
    except FileNotFoundError:
        print(f"오류: train 폴더를 찾을 수 없습니다: {train_dir}")
        sys.exit()


# 함수: 모델 구조 선언, 학습된 가중치 로드
def get_model(model_path, num_classes):
    # (1) 모델 뼈대 생성
    model = models.resnet50(weights=None)

    # (2) train과 동일하게 모델 구조 수정
    num_frts = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_frts, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    # (3) train.py에서 만들어낸 가중치 로드
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval() # eval: dropout, BN 비활성화
    return model

# 함수: 사용자 입력 이미지를 모델에 넣을 수 있게 전처리
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')  # 이미지 로드
    image = transform(image).unsqueeze(0)  # 배치 차원 추가 (n개짜리 이미지임을 명시하는 차원. 우리 프로젝트의 경우 1개 이미지)
    return image.to(device)

# 함수: 병명을 예측
def predict_disease(model, image_path):
    input_tensor = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(input_tensor)

        # 1. Softmax 함수를 통과시켜 확률값으로 변환
        probs = torch.nn.functional.softmax(outputs, dim=1)

        # 2. 가장 높은 확률값과 그 인덱스(top_class) 추출
        top_prob, top_class = probs.topk(1, dim=1)

        # 3. 값 꺼내기
        confidence = top_prob.item() * 100  # %로 변환
        predicted_index = top_class.item()

    # 결과 딕셔너리 생성
    result = {
        "class_name": class_names[predicted_index],
        "confidence": confidence
    }

    return result



# test
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    MODEL_PATH = os.path.join(project_root, 'training', 'best_resnet50_20251127_2211.pth')
    TEST_IMAGE_PATH = os.path.join(project_root, 'test_image', 'test.jpg')
    TRAIN_DIR = os.path.join(project_root, 'training', 'dataset', 'train')
    print(f"가중치 파일 경로 확인: {MODEL_PATH}")

    try:
        # 모델 로드
        class_names = load_class_names(TRAIN_DIR)
        if os.path.exists(MODEL_PATH):
            model = get_model(MODEL_PATH, len(class_names))
            print("모델 로드 성공")



            

        else:
            print("오류: 가중치 파일이 경로에 없습니다!")
            exit()

        # 예측 실행 (테스트 이미지가 실제로 있을 때만)
        if os.path.exists(TEST_IMAGE_PATH):
            result = predict_disease(model, TEST_IMAGE_PATH)
            print(f"진단 결과: {result}")
        else:
            print(f"알림: 테스트 이미지를 찾을 수 없습니다 ({TEST_IMAGE_PATH})")
            print("경로가 맞는지 확인하거나 이미지를 해당 위치에 넣어주세요.")

    except Exception as e:
        print(f"에러 발생: {e}")

 