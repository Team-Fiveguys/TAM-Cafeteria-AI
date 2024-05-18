from flask import Flask, request, jsonify
from joblib import load
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

app = Flask(__name__)

# joblib 파일로부터 모델 불러오기
model = load('tam.joblib')

#데이터 전처리 후 예측하는 함수
def preprocess_and_predict(new_data):
    # 저장된 모델 불러오기
    ridge_pipeline = joblib.load('tam.joblib')

    # 새로운 데이터 준비 (예: new_data)
    # 새로운 데이터는 반드시 저장된 모델을 학습할 때 사용했던 컬럼과 동일한 컬럼을 가져야 합니다.

    # 예측 수행
    predictions = ridge_pipeline.predict(new_data)

    print("새로운 데이터에 대한 예측 결과:", predictions)  

    return predictions

# JSON 데이터를 DataFrame으로 변환하는 함수로 변경
def convert_json_to_dataframe(json_data):
    # JSON 데이터를 DataFrame으로 변환
    data_df = pd.DataFrame([json_data])
    return data_df


@app.route('/')
def hello_world():
  return 'Hello,  My name is EOM. I have BIG muscle. I am strong man HA!HA!HA!'

@app.route('/predict', methods=['POST'])
def predict():
    # 클라이언트로부터 변수를 받습니다.
    data = request.get_json(force=True)
    #json데이터를 리스트 형태로 변환
    dataframe_data = convert_json_to_dataframe(data)
    #리스트 데이터를 모델에 입력하고 예측값 반환
    predict_result = preprocess_and_predict(dataframe_data)

     # 예측 결과를 클라이언트에게 반환 (NumPy 배열을 리스트로 변환)
    return jsonify(predict_result=predict_result.tolist())


if __name__ == '__main__':
    app.run(debug=True)
