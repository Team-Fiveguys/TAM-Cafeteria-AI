from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

# joblib 파일로부터 모델 불러오기
model = load('tam.joblib')

#클라이언트에게 받은 변수 모델에 맞는 형식으로 바꾸는 함수
def model_parameter(input):
    #TODO:로직구현해야함
    return 0

# 예측 모델 함수
def predict_model(data1, data2):
    result = model.predict([[data1, data2]])
    return result

@app.route('/')
def hello_world():
  return 'Hello,  My name is EOM. I have BIG muscle. I am strong man HA!HA!HA!'

@app.route('/predict', methods=['POST'])
def predict():
    # 클라이언트로부터 변수를 받습니다.
    data = request.get_json(force=True)
    data1 = data['변수1']
    data2 = data['변수2']

    # 예측 모델 함수에 변수를 넘겨줍니다.
    predict_result = predict_model(data1, data2)

    # 예측 결과를 클라이언트에게 반환합니다.
    return jsonify(predict_result=predict_result)

if __name__ == '__main__':
    app.run(debug=True)
