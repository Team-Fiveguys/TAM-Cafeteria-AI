from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

app = Flask(__name__)

#데이터 전처리 후 예측하는 함수
def preprocess_and_predict(new_data):
    """
    새로운 데이터에 대해 동일한 전처리 과정을 적용하고, 모델을 사용하여 예측을 수행하는 함수입니다.

    Parameters:
    new_data (pd.DataFrame): 예측을 수행할 새로운 데이터

    Returns:
    np.array: 모델에 의해 예측된 결과 값
    """

    # 카테고리 변수와 수치형 변수 분리
    categorical_features = ['요일', '분류메뉴', '학관분류메뉴']
    numeric_features = new_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # 카테고리 데이터를 위한 변환기
    categorical_transformer = OneHotEncoder()

    # 수치형 데이터를 위한 변환기
    numeric_transformer = MinMaxScaler()

    # ColumnTransformer를 사용하여 수치형 데이터와 카테고리 데이터에 다른 변환을 적용
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 파이프라인 구축: 전처리 단계와 모델을 연결
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    # 파이프라인을 사용하여 새로운 데이터에 대한 예측 수행
    y_pred = pipeline.predict(new_data)

    return y_pred

#json 데이터를 리스트로 변경하는 함수
def convert_json_to_list(json_data):
    # JSON 데이터의 값들만 리스트로 추출
    data_list = list(json_data.values())
    return data_list


# joblib 파일로부터 모델 불러오기
model = load('tam.joblib')

@app.route('/')
def hello_world():
  return 'Hello,  My name is EOM. I have BIG muscle. I am strong man HA!HA!HA!'

@app.route('/predict', methods=['POST'])
def predict():
    # 클라이언트로부터 변수를 받습니다.
    data = request.get_json(force=True)
    #json데이터를 리스트 형태로 변환
    list_data = convert_json_to_list(data)
    #리스트 데이터를 모델에 입력하고 예측값 반환
    predict_result = preprocess_and_predict(list_data)

    # 예측 결과를 클라이언트에게 반환
    return jsonify(predict_result=predict_result)

if __name__ == '__main__':
    app.run(debug=True)
