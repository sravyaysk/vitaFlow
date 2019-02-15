#!flask/bin/python
from flask import Flask, render_template, jsonify, send_file, request


import os
import base64
from pprint import pprint
import pickle

import config
import views
import annotate
import cropper


app = Flask(__name__, static_folder='static', static_url_path='/static')

sample_data = {"url": "/static/data/images/pexels-photo-60091.jpg",
               "id": "pexels-photo-60091.jpg",
               "folder": "collection_01/part_1",
               "annotations": [
                   {"tag": "Eagle", "x": 475, "y": 225, "width": 230.555555554, "height": 438.888888886}
               ]
               }


@app.route('/inc/validateTagsAndRegions.php', methods=['POST', 'GET'])
def _rest_validate_tags_and_regions():
    form_data = dict(request.form)
    # pprint(form_data)
    if 'sendInfo' in form_data.keys():
        annotate.validate_tags_and_regions(request.form)
    return _rest_get_new_image()


@app.route('/inc/getNewImage.php', methods=['POST', 'GET'])
def _rest_get_new_image():
    views.GetNewImage.refresh()
    return jsonify(views.GetNewImage.get_new_image())
#
# @app.route('/data/<path:path>')
# def _rest_annotate_image(path=''):
#     print('Path is {}'.format(path))
#     return send_file('')



@app.route('/annotate_image')
def annotate_image():
    return render_template('index.html')


@app.route('/')
@app.route('/<image>')
def annotate_image2(image=None):
    if image:
        print('--'* 54)
        print(image)
    return render_template('index.html')



@app.route('/review_annotation')
# @app.route('/review_annotation/<image:image>')
def review_annotation():
    return render_template('index.html')


@app.route('/show_completed_images')
def show_completed_images():
    # Get data & show
    # show data nicely
    views.GetNewImage.refresh()
    # print(views.GetNewImage.PendingImages)
    return jsonify(views.GetNewImage.CompletedImages)


def show_pending_images():
    pass


def show_all_images():
    pass


def show_summary():
    pass


def login_logout():
    pass


@app.route("/cropper/<image_name>")
def page_cropper(image_name=None):
    data = {'image_name': "/" + os.path.join(config.IMAGE_ROOT_DIR, image_name)}
    return render_template("Cropper_js.html", data=data)


@app.route("/cropper/upload.php", methods=['POST'])
@app.route("/upload.php", methods=['POST'])
def cropper_upload():
    data = dict(request.form)
    return cropper.cropper_upload(data)


if __name__ == '__main__':
    app.run(debug=True)
