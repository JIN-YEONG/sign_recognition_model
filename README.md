# sign_recognition_model

비트캠프 최종 프로젝트에서 사용한 모델 (19.09 ~ 19.11)
  - 간판의 위치를 예측하는 모델
  - 간판의 종류를 분류하는 모델
  
  
데이터의 경우 용량이 크기 때문에 훈련된 모델과 훈련코드, 결과 확인 코드만 등록


input_data = iamge_path

output_data
 - location_model : [x, y, w, h] (객체의 중심점과 박스의 가로세로 길이)
 - categorical_model : one-hot 인코딩된 라벨 값
