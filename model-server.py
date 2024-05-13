from flask import Flask, request, jsonify

app = Flask(__name__)

# 예시 예측 모델 함수
def predict_model(something):
    # 여기 모델 구현 or 연결
    result = something
    return result

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
