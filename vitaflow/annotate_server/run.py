#!flask/bin/python
from flask import Flask, render_template, jsonify, send_file, request

from pprint import pprint

app = Flask(__name__, static_folder='static', static_url_path='/static')


@app.route('/test')
def index():
    return "Hello, World!"


@app.route('/inc/validateTagsAndRegions.php', methods = ['POST', 'GET'])
def validateTagsAndRegions():
    pprint(request.__dict__)
    return jsonify({"url": "/static/data/images/collection_01/part_1/pexels-photo-60091.jpg",
                    "id": "pexels-photo-60091.jpg",
                    "folder": "collection_01/part_1", "annotations": []})


@app.route('/inc/getNewImage.php', methods = ['POST', 'GET'])
def getNewImage():
    pprint(request.__dict__)

    return jsonify({"url": "/static/data/images/collection_01/part_1/pexels-photo-60091.jpg",
                    "id": "pexels-photo-60091.jpg",
                    "folder": "collection_01/part_1", "annotations": []})


@app.route('/data/<path:path>')
def annotate_image(path=''):
    print('Path is {}'.format(path))
    return send_file('')


@app.route('/')
def home_page():
    return render_template('index.html')





if __name__ == '__main__':
    app.run(debug=True)
