from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

app = Flask(__name__)

def preprocess_and_predict(new_data):
    try:
        # 저장된 모델 불러오기
        ridge_pipeline = joblib.load('tam.joblib')

        # 예측 수행
        predictions = ridge_pipeline.predict(new_data)

        print("새로운 데이터에 대한 예측 결과:", predictions)  

        return predictions
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        raise

def convert_json_to_dataframe(json_data):
    try:
        # JSON 데이터를 DataFrame으로 변환
        data_df = pd.DataFrame([json_data])
        return data_df
    except Exception as e:
        print(f"데이터 변환 중 오류 발생: {e}")
        raise

@app.route('/')
def hello_world():
    return 'Hello,  My name is EOM. I have BIG muscle. I am strong man HA!HA!HA!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 클라이언트로부터 변수를 받습니다.
        data = request.get_json(force=True)
        print(data)
        #json데이터를 리스트 형태로 변환
        dataframe_data = convert_json_to_dataframe(data)
        print(dataframe_data)
        #리스트 데이터를 모델에 입력하고 예측값 반환
        predict_result = preprocess_and_predict(dataframe_data)
        print(predict_result)

        # 예측 결과를 클라이언트에게 반환
        return jsonify(predict_result=predict_result.tolist())
    except Exception as e:
        print(f"예측 요청 처리 중 오류 발생: {e}")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
