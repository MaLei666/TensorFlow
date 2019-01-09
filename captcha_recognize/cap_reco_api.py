#!/home/zkfr/.local/share/virtualenvs/xf-5EfV3Nly/bin/python
#-*- coding:utf-8 -*-
# @author : MaLei 
# @datetime : 2019-01-09 21:58
# @file : cap_reco_api.py
# @software : PyCharm

import sys
from werkzeug.wrappers import Response
from flask import Flask, jsonify, request
from captcha_recognize import run_predict
sys.path.append('../')
app = Flask(__name__)

class JsonResponse(Response):

    @classmethod
    def force_type(cls, response, environ=None):
        if isinstance(response, (dict, list)):
            response = jsonify(response)
        return super(JsonResponse, cls).force_type(response, environ)

app.response_class = JsonResponse

@app.route('/get/')
def get():
    captcha = run_predict('./data/test_data/1ab2s_num286.jpg')
    return captcha if captcha else 'no captcha!'

def run():
    app.run(host='0.0.0.0', port=8001)

if __name__ == '__main__':
    run()