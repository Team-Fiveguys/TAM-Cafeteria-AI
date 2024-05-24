from flask import Flask, request, jsonify
import joblib
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pymysql
import pathlib
import textwrap
import google.generativeai as genai
from datetime import datetime, date, timedelta
import holidays
import json


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
def get_similar_category(menu_name, cafeteria_id):
    model = genai.GenerativeModel('gemini-pro')  # 모델 선택
    
    if cafeteria_id == 1:
        response = model.generate_content(f"{menu_name}이 메뉴가 ['육류', '국밥', '돈까스', '전골류', '찌개', '국수', '비빔밥', '맑은국'] 이 메뉴 카테고리 중에서 가장 비슷한 카테고리를 한단어로 말해줘. 카테고리에 있는 것만 대답해.")
        
    elif cafeteria_id == 2:
        response = model.generate_content(f"{menu_name}이 메뉴가 ['육류', '밥류', '돈까스', '비빔밥', '분식류', '면류', '찌개류', '연어', '국밥', '볶음밥'] 이 메뉴 카테고리 중에서 가장 비슷한 카테고리를 한단어로 말해줘. 카테고리에 있는 것만 대답해.")
        
    result_text = response.text.strip()
    return result_text

#날짜와 식당에 해당하는 메뉴 이름을 데이터베이스에서 가져옵니다.
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
    date_object = datetime.strptime(date_string, '%Y-%m-%d')
    # 요일을 숫자로 반환 (월요일=0, 일요일=6)
    weekday_number = date_object.weekday()
    # 숫자를 요일 이름으로 변환
    days = ["월", "화", "수", "목", "금", "토", "일"]
    return days[weekday_number]

# 날짜와 학기번호 입력받아 주차를 계산하는 함수
def calculate_week(local_date_str, semester_number):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # semester_number에 해당하는 start_date 가져오기
            sql_semester = "SELECT start_date FROM semester WHERE semester_num = %s"
            cursor.execute(sql_semester, (semester_number,))
            start_date = cursor.fetchone()
            if start_date:
                start_date = start_date[0]
                # start_date가 datetime.date 객체일 경우 문자열로 변환
                if isinstance(start_date, date):
                    start_date = start_date.strftime('%Y-%m-%d')
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                
                local_date = datetime.strptime(local_date_str, "%Y-%m-%d")
                week_number = (local_date - start_date).days // 7 + 1
                return week_number
    finally:
        connection.close()
    return 0

#전주 식수를 불러오는 함수
def get_previous_week_headcount(local_date_str):
    # 문자열 형식의 날짜를 datetime 객체로 변환
    local_date = datetime.strptime(local_date_str, "%Y-%m-%d")
    # 일주일 전 날짜 계산
    previous_week_date = local_date - timedelta(days=7)
    # 날짜를 문자열 형식으로 다시 변환
    previous_week_date_str = previous_week_date.strftime("%Y-%m-%d")
    
    # 데이터베이스 연결
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 일주일 전 날짜에 해당하는 count1 값을 조회하는 쿼리
            query = """
                    SELECT count1 
                    FROM headcount_data 
                    WHERE date = %s
                    """
            cursor.execute(query, (previous_week_date_str,))
            result = cursor.fetchone()
            if result:
                return result[0]
            else:
                return None
    finally:
        connection.close()

#학기번호 계산을 위한 함수
def calculate_semester_number(local_date_str):
    start_date = datetime(2023, 3, 1)  # 기준 시작 날짜
    local_date = datetime.strptime(local_date_str, "%Y-%m-%d")  # 입력받은 날짜
    
    # 시작 날짜와 입력 날짜의 차이(달 기준)
    month_diff = (local_date.year - start_date.year) * 12 + (local_date.month - start_date.month)
    
    # 학기 번호 계산
    # 6개월마다 학기 번호가 1씩 증가
    semester_number = 1 + month_diff // 6
    
    return semester_number

#json데이터를 모델에 입력하기 전 필요한 변수들을 채워주는 함수
def fill_data(data):
    # 명진당 평균 식수 딕셔너리
    day_avg_meals1 = {
        "월": 495,
        "화": 505,
        "수": 501,
        "목": 462,
        "금": 261
    }
    # 학관 평균 식수 딕셔너리
    day_avg_meals2 = {
        "월": 522,
        "화": 528,
        "수": 495,
        "목": 435,
        "금": 218,
        "토": 150,
        "일": 154
    }

    # 요일에 따른 평균 식수 값을 가져옴.
    avg_meals1 = day_avg_meals1.get(get_weekday(data.get("local_date")))
    avg_meals2 = day_avg_meals2.get(get_weekday(data.get("local_date")))
    # local_date 가져오기
    local_date_str = data.get("local_date")
    local_date = datetime.strptime(local_date_str, "%Y-%m-%d")
    # 학기 번호 계산
    semester_number = calculate_semester_number(local_date_str)
    # 주차 계산
    week = calculate_week(local_date_str, semester_number)
    
    exam_period = 1 if week in [7, 8, 14, 15] else 0

    # 한국 공휴일 설정
    kr_holidays = holidays.KR()
    # local_date 가져오기
    local_date = datetime.strptime(data.get("local_date"), "%Y-%m-%d")
    # 휴일 전날, 휴일 다음날, 연휴 전날, 연휴 다음날 설정
    prev_day = local_date - timedelta(days=1)
    next_day = local_date + timedelta(days=1)
    day_before_prev = prev_day - timedelta(days=1)
    day_after_next = next_day + timedelta(days=1)

    is_local_date_holiday = local_date in kr_holidays
    is_prev_day_holiday = prev_day in kr_holidays
    is_next_day_holiday = next_day in kr_holidays
    is_day_before_prev_holiday = day_before_prev in kr_holidays
    is_day_after_next_holiday = day_after_next in kr_holidays
    
    weekday = get_weekday(data.get("local_date"))

    cafeteria_id = data.get("cafeteria_id")

    # cafeteria_id에 따라 반환할 데이터 구성을 변경
    if cafeteria_id == 1:
        filled_data = {
            "요일": weekday,
            "전주 식수": get_previous_week_headcount(data.get("local_date")),
            "요일 평균 식수": avg_meals1,
            "시험기간": exam_period,
            "축제유무": data.get("festival", 0),
            "간식배부": data.get("snack", 0),
            "예비군유무": data.get("reservist", 0),
            "분류메뉴": get_similar_category(get_menu_name(data.get("local_date"), 1), 1), #명진당 id:1
            "주차": week,
            "학관분류메뉴": get_similar_category(get_menu_name(data.get("local_date"), 2), 2), #학관 id:2
            "방학유무": 0,
            "개강주": 1 if week == 1 else 0,
            "종강주": 1 if week == 15 else 0,
            "시험끝목금": 1 if exam_period == 1 and weekday in ["목", "금"] else 0,
            "공휴일유무":  1 if is_local_date_holiday else 0,
            "학기번호": calculate_semester_number(data.get("local_date")),
            "휴일 전날": 1 if is_prev_day_holiday else 0,
            "휴일 다음날": 1 if is_next_day_holiday else 0,
            "연휴 전날": 1 if is_day_before_prev_holiday else 0,
            "연휴 다음날": 1 if is_day_after_next_holiday else 0,
            "매움": data.get("spicy", 0)
        }
        return filled_data
    elif cafeteria_id == 2:
        filled_data = {
            "학관분류메뉴": get_similar_category(get_menu_name(data.get("local_date"), 2), 2), #학관 id:2
            "요일": weekday,
            "전주 식수": get_previous_week_headcount(data.get("local_date")),
            "요일 평균 식수": avg_meals2,
            "명진당분류메뉴": get_similar_category(get_menu_name(data.get("local_date"), 1), 1), #명진당 id:1
            "학기번호": calculate_semester_number(data.get("local_date")),
            "휴일 전날": 1 if is_prev_day_holiday else 0,
            "휴일 다음날": 1 if is_next_day_holiday else 0,
        }
               
        return filled_data
    else:
        # 잘못된 cafeteria_id가 주어진 경우를 처리
        raise ValueError("Invalid cafeteria_id")

#예측 모델을 불러와 예측을 수행하는 함수
def load_and_predict(new_data, cafeteria_id):
    try:
        # 저장된 모델 불러오기
        if cafeteria_id == 1:
            predict_model = joblib.load('tam1.joblib')
            predictions = predict_model.predict(new_data)
        elif cafeteria_id == 2:
            with open('tam2.pkl', 'rb') as file:
                predict_model = joblib.load(file)
                predictions = predict_model.predict(new_data)
        
        print("새로운 데이터에 대한 예측 결과:", predictions)  
        print(new_data)
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

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'UMP is strong man.'

#서버 체크
@app.route('/health', methods=['GET'])
def heatlh():
    return jsonify({'status': 'UP'}), 200


@app.route('/predict1', methods=['POST'])
def predict1():
    conn = None
    try:
        # 클라이언트로부터 변수를 받습니다.
        data = request.get_json(force=True)
        # 클라이언트가 입력한 데이터 외의 필요한 데이터 채우기
        filled_data = fill_data(data)
        # json 데이터를 리스트 형태로 변환
        dataframe_data = convert_json_to_dataframe(filled_data)
        # 리스트 데이터를 모델에 입력하고 예측값 반환
        predict_result = load_and_predict(dataframe_data, 1)
        predict_result_int = int(predict_result[0])
        # 데이터베이스 연결
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # JSON 데이터와 예측 결과를 predict 테이블에 저장
            insert_sql = """
                INSERT INTO predict (date, cafeteria_id, data, predict_result) 
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                data = VALUES(data), predict_result = VALUES(predict_result)
            """
            cursor.execute(insert_sql, (
                data.get("local_date"), 
                data.get("cafeteria_id"), 
                json.dumps(data), 
                predict_result_int
            ))
            conn.commit()
        # 예측 결과를 클라이언트에게 반환
        return jsonify(predict_result=predict_result_int)
    except Exception as e:
        print(f"예측 요청 처리 중 오류 발생: {e}")
        return jsonify(error=str(e)), 500
    finally:
        # 데이터베이스 연결 종료
        if conn is not None:
            conn.close()

#학생회관 식수 예측 및 변수,결과 저장
@app.route('/predict2', methods=['POST'])
def predict2():
    try:
        conn = None
        # 클라이언트로부터 변수를 받습니다.
        data = request.get_json(force=True)
        #클라이언트가 입력한 데이터 외의 필요한 데이터 채우기
        filled_data = fill_data(data)
        #json데이터를 리스트 형태로 변환
        dataframe_data = convert_json_to_dataframe(filled_data)
        #리스트 데이터를 모델에 입력하고 예측값 반환
        predict_result = load_and_predict(dataframe_data, 2) #학관 예측 모델을 불러오는 함수로 바꿔야 함.
        predict_result_int = int(predict_result[0])
        # 데이터베이스 연결
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # JSON 데이터와 예측 결과를 predict 테이블에 저장
            insert_sql = """
                INSERT INTO predict (date, cafeteria_id, data, predict_result) 
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                data = VALUES(data), predict_result = VALUES(predict_result)
            """
            cursor.execute(insert_sql, (
                data.get("local_date"), 
                data.get("cafeteria_id"), 
                json.dumps(data), 
                predict_result_int
            ))
            conn.commit()
        # 예측 결과를 클라이언트에게 반환
        return jsonify(predict_result=predict_result_int)
    except Exception as e:
        print(f"예측 요청 처리 중 오류 발생: {e}")
        return jsonify(error=str(e)), 500
    finally:
        # 데이터베이스 연결 종료
        if conn is not None:
            conn.close()

#개강일 추가    
@app.route('/add_semester', methods=['POST'])
def add_semester():
    try:
        # 클라이언트로부터 변수를 받습니다.
        data = request.get_json(force=True)
        start_date = data.get('start_date')
        semester_num = calculate_semester_number(data.get("start_date"))

        # 데이터베이스 연결
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # SQL 쿼리 작성
        query = """
        INSERT INTO semester (start_date, semester_num)
        VALUES (%s, %s)
        """
        cursor.execute(query, (start_date, semester_num))
        connection.commit()

        # 연결 종료
        cursor.close()
        connection.close()

        return jsonify(message="Semester data added successfully."), 200
    except Exception as e:
        print(f"학기 데이터 추가 중 오류 발생: {e}")
        return jsonify(error=str(e)), 500

#개강일 조회
@app.route('/start_date', methods=['GET'])
def get_start_date():
    try:
        # 데이터베이스 연결
        connection = get_db_connection()
        cursor = connection.cursor()

        # SQL 쿼리 작성
        query = "SELECT start_date FROM semester LIMIT 1"
        cursor.execute(query)
        result = cursor.fetchone()

        # 결과를 리스트로 변환 및 YYYY-MM-DD 형식으로 포맷팅
        start_date = result[0].strftime('%Y-%m-%d') if result else None  # 결과가 있으면 포맷팅, 없으면 None

        # 연결 종료
        cursor.close()
        connection.close()

        # JSON 응답 반환
        return jsonify(start_date=start_date), 200
    except Exception as e:
        print(f"start_date 조회 중 오류 발생: {e}")
        return jsonify(error=str(e)), 500

#실제 식수 저장
@app.route('/headcount', methods=['POST'])
def post_headcount():
    # 클라이언트로부터 데이터를 받음
    data = request.get_json()
    local_date = data.get('local_date')
    cafeteria_id = data.get('cafeteria_id')
    headcount = data.get('headcount')

    # 데이터베이스 연결
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 입력 받은 local_date에 해당하는 row가 있는지 확인
            sql = "SELECT EXISTS(SELECT * FROM headcount_data WHERE date = %s)"
            cursor.execute(sql, (local_date,))
            exists = cursor.fetchone()[0]

            if exists:
                # cafeteria_id에 따라 count1 또는 count2에 headcount를 저장
                if cafeteria_id == 1:
                    update_sql = "UPDATE headcount_data SET count1 = %s WHERE date = %s"
                elif cafeteria_id == 2:
                    update_sql = "UPDATE headcount_data SET count2 = %s WHERE date = %s"
                cursor.execute(update_sql, (headcount, local_date))
            else:
                # 해당 날짜에 대한 row가 없으면 새로운 row를 생성
                if cafeteria_id == 1:
                    insert_sql = "INSERT INTO headcount_data (date, count1) VALUES (%s, %s)"
                elif cafeteria_id == 2:
                    insert_sql = "INSERT INTO headcount_data (date, count2) VALUES (%s, %s)"
                cursor.execute(insert_sql, (local_date, headcount))

            conn.commit()
    finally:
        conn.close()

    return jsonify({"message": "Headcount data saved successfully"}), 200

#실제 식수 조회
@app.route('/headcount', methods=['GET'])
def get_headcount():
    # 쿼리 파라미터로부터 데이터를 받음
    local_date = request.args.get('local_date')
    cafeteria_id = request.args.get('cafeteria_id', type=int)

    # 데이터베이스 연결
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # cafeteria_id에 따라 적절한 count 값을 조회
            if cafeteria_id == 1:
                query_sql = "SELECT count1 FROM headcount_data WHERE date = %s"
            elif cafeteria_id == 2:
                query_sql = "SELECT count2 FROM headcount_data WHERE date = %s"
            else:
                return jsonify({"error": "Invalid cafeteria_id"}), 400

            cursor.execute(query_sql, (local_date,))
            row = cursor.fetchone()
            
            # 조회된 데이터가 없는 경우
            if row is None:
                return jsonify({"error": "Data not found"}), 404

            # 조회된 데이터가 있는 경우
            data = {
                "headcount": row[0]  # count1 또는 count2
            }

            return jsonify(data), 200
    finally:
        conn.close()

#예측 변수 및 결과 조회
@app.route('/predict', methods=['GET'])
def get_predict_data():
    # 쿼리 파라미터로부터 데이터를 받음
    local_date = request.args.get('local_date')
    cafeteria_id = request.args.get('cafeteria_id', type=int)

    # 데이터베이스 연결
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            query_sql = """
                SELECT data, predict_result 
                FROM predict 
                WHERE date = %s AND cafeteria_id = %s
            """
            cursor.execute(query_sql, (local_date, cafeteria_id))
            row = cursor.fetchone()
            
            # 조회된 데이터가 없는 경우
            if row is None:
                return jsonify({"error": "Data not found"}), 404

            # 조회된 데이터가 있는 경우
            data = {
                "data": row[0],
                "predict_result": row[1]
            }

            return jsonify(data), 200
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)
