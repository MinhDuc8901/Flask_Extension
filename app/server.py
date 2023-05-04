from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request, jsonify

from prediction.prediction import URL_Converter,predict_url_toxics
from globle import PATH_DIR_URL, FILE_NAME_URL
from dao.response import response_data

import joblib


# khởi tạo Flask server backend
app = Flask(__name__)


# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#Load model 
def load_model_url():
    array_model = []
    for index in range(1,6):
        model = joblib.load(PATH_DIR_URL+str(index)+FILE_NAME_URL)
        array_model.append(model)
    
    return array_model

@app.route('/check_url',methods=['POST'])
@cross_origin(origins='*')
def check_url():
    try:
        get_data_client = request.json
        urls = get_data_client['urls']
        print(urls)
        # urls = URL_Converter(urls)
        results = predict_url_toxics(urls,models)
        print("URL: "+urls)
        print("Kết quả trả về: "+ str(results))
        return response_data(200,"Thành công",results)
    except :
        return response_data(400,"Không thành công",[])


models = load_model_url()
if __name__ == '__main__':
    app.run(host='0.0.0.0',port='9999')
    