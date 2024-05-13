from waitress import serve
from model_app import app

serve(app, host='0.0.0.0', port=8080)
