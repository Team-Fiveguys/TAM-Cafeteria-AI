from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pymysql
import pathlib
import textwrap
import google.generativeai as genai
import datetime


# RDS 인스턴스 정보
endpoint = 'tamdb.cfk0ie0wc7k6.ap-northeast-2.rds.amazonaws.com'
username = 'root'
password = 'rang60192185'
database_name = 'tam'
# Google API 키를 설정합니다.
GOOGLE_API_KEY = 'AIzaSyC9MEXmFms89O1p9nrX2Mb53rYg-ua7k54'
genai.configure(api_key=GOOGLE_API_KEY)

#데이터베이스 연결 반환
def get_db_connection():
    connection = pymysql.connect(host=endpoint, user=username, passwd=password, db=database_name)
    return connection

#메뉴이름 입력 받아 카테고리(메뉴분류) 반환
def get_similar_category(menu_name):
    model = genai.GenerativeModel('gemini-pro')  # 모델 선택
    response = model.generate_content(f"{menu_name}이 메뉴가 ['육류', '국밥', '돈까스', '전골류', '찌개', '국수', '비빔밥', '맑은국'] 이 메뉴 카테고리 중에서 가장 비슷한 카테고리를 한단어로 말해줘. 카테고리에 있는 것만 대답해.")
    
    # 결과 텍스트를 Markdown 형식으로 변환합니다.
    result_text = textwrap.indent(response.text, '> ', predicate=lambda _: True)
    
    return result_text

#날짜와 식당에 해당하는 분류메뉴 이름을 데이터베이스에서 가져옵니다.
def get_menu_name(local_date, cafeteria_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # local_date와 cafeteria_id에 해당하는 diet id를 가져옴
            sql_diet = "SELECT id FROM diet WHERE local_date = %s AND cafeteria_id = %s"
            cursor.execute(sql_diet, (local_date, cafeteria_id))
            diet_id = cursor.fetchone()
            if diet_id:
                diet_id = diet_id[0]

                # diet_id에 해당하는 menu_id들을 menu_diet에서 가져옴
                sql_menu_diet = "SELECT menu_id FROM menu_diet WHERE diet_id = %s"
                cursor.execute(sql_menu_diet, (diet_id,))
                menu_ids = cursor.fetchall()

                if menu_ids:
                    menu_ids = [menu_id[0] for menu_id in menu_ids]

                    # 첫 번째 menu_id에 해당하는 메뉴 이름을 menu 테이블에서 가져옴
                    sql_menu = "SELECT name FROM menu WHERE id = %s"
                    cursor.execute(sql_menu, (menu_ids[0],))
                    menu_name = cursor.fetchone()
                    if menu_name:
                        return menu_name[0]
    finally:
        connection.close()
    return None

#날짜 받아 요일 반환
def get_weekday(date_string):
    date_object = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    # 요일을 숫자로 반환 (월요일=0, 일요일=6)
    weekday_number = date_object.weekday()
    # 숫자를 요일 이름으로 변환
    days = ["월", "화", "수", "목", "금", "토", "일"]
    return days[weekday_number]

#json데이터를 모델에 입력하기 전 필요한 변수들을 채워주는 함수
def fill_data(data):
    # "요일"에 따른 "요일 평균 식수"를 정의하는 딕셔너리
    day_avg_meals = {
        "월": 495,
        "화": 505,
        "수": 501,
        "목": 462,
        "금": 261,
    }
    # 요일에 따른 평균 식수 값을 가져옴.
    avg_meals = day_avg_meals.get(data.get("요일"), 500)

    filled_data = {
        "요일": get_weekday(data.get("local_date")),
        "전주 식수": 500,
        "요일 평균 식수": avg_meals,
        "시험기간": data.get("exam", 0),
        "축제유무": data.get("festival", 0),
        "간식배부": data.get("snack", 0),
        "예비군유무": data.get("reservist", 0),
        "분류메뉴": get_similar_category(get_menu_name(data.get("local_date"), data.get("cafeteria_id"))),
        "주차": 11, #localdate에서 계산
        "학관분류메뉴": get_similar_category(get_menu_name(data.get("local_date"), 2)), #학관 id=2
        "방학유무": data.get("vacation", 0),
        "개강주": 0, #주차에서 계산
        "종강주": 0, #주차에서 계산
        "시험끝목금": 0, 
        "공휴일유무": data.get("holiday", 0),
        "학기번호": 3,
        "휴일 전날": 0,
        "휴일 다음날": 0,
        "연휴 전날": 0,
        "연휴 다음날": 0,
        "매움": data.get("spicy", 0)
    }
    return filled_data

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

# JSON 데이터를 DataFrame으로 변환하는 함수
def convert_json_to_dataframe(json_data):
    try:
        data_df = pd.DataFrame([json_data])
        return data_df
    except Exception as e:
        print(f"데이터 변환 중 오류 발생: {e}")
        raise

@app.route('/')
def hello_world():
    return 'UMP is strong man.'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 클라이언트로부터 변수를 받습니다.
        data = request.get_json(force=True)
        #클라이언트가 입력한 데이터 외의 필요한 데이터 채우기
        filled_data = fill_data(data)
        #json데이터를 리스트 형태로 변환
        dataframe_data = convert_json_to_dataframe(filled_data)
        #리스트 데이터를 모델에 입력하고 예측값 반환
        predict_result = preprocess_and_predict(dataframe_data)
        # 예측 결과를 클라이언트에게 반환
        return jsonify(predict_result=predict_result.tolist())
    except Exception as e:
        print(f"예측 요청 처리 중 오류 발생: {e}")
        return jsonify(error=str(e)), 500
    
@app.route('/predict_test', methods=['POST'])
def test():
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

@app.route('/find_category', methods=['POST'])
def find_category():
    """클라이언트로부터 메뉴 이름을 받아 가장 비슷한 카테고리를 찾아 반환합니다."""
    data = request.get_json(force=True)
    menu_name = data.get('menu_name')
    
    if not menu_name:
        return jsonify(error="menu_name is required"), 400
    
    try:
        similar_category = get_similar_category(menu_name)
        return jsonify(similar_category=similar_category)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
