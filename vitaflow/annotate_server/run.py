#!flask/bin/python
import os

from flask import Flask, render_template, jsonify, request, Markup

import annotate
import config
import cropper
import image_manager
import stats

image_manager.GetNewImage.refresh()

app = Flask(__name__, static_folder='static', static_url_path='/static')

sample_data = {"url": "static/images/NoImage.png",
               "id": "NoImage.png",
               "folder": "collection_01/part_1",
               "annotations": [
                   {"tag": "Eagle", "x": 475, "y": 225, "width": 230.555555554, "height": 438.888888886}
               ]
               }


@app.route('/inc/validateTagsAndRegions', methods=['POST', 'GET'])
def _rest_validate_tags_and_regions():
    form_data = dict(request.form)
    # pprint(form_data)
    if 'sendInfo' in form_data.keys():
        annotate.validate_tags_and_regions(request.form)
    return _rest_get_new_image()


@app.route('/inc/getNewImage', methods=['POST', 'GET'])
@app.route('/inc/getNewImage/<image>', methods=['POST', 'GET'])
def _rest_get_new_image(image=None):
    if image:
        print('\n' + '=' * 54 + image + '\n' + '=' * 54)
        try:
            return jsonify(image_manager.GetNewImage.get_specific_image(image))
        except Exception as err:
            print("==" * 15)
            print(err)
            print(image)
            print(locals())
    image_manager.GetNewImage.refresh()
    return jsonify(image_manager.GetNewImage.get_new_image())


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
        print('\n' + '-' * 54 + image + '\n' + '-' * 54)
    return render_template('index.html')


@app.route('/review_annotation')
# @app.route('/review_annotation/<image:image>')
def review_annotation():
    return render_template('index.html')


@app.route('/show_completed_images')
def show_completed_images():
    # Get data & show
    # show data nicely
    image_manager.GetNewImage.refresh()
    # print(image_manager.GetNewImage.PendingImages)
    return jsonify(image_manager.GetNewImage.completed_images)


def show_pending_images():
    pass


@app.route('/show_all_images')
def show_all_images():
    return jsonify([
        ('receipt_images', list(image_manager.GetNewImage.receipt_images.keys())),
        ('pending_images', list(image_manager.GetNewImage.pending_images)),
        ('completed_images', list(image_manager.GetNewImage.completed_images)),
    ])


@app.route('/summary')
@app.route('/summary/')
def show_summary():
    data = list(image_manager.GetNewImage.annotated_files.keys())
    print('Request to display {} Records'.format(len(data)))
    return render_template('summary.html', data=data)


@app.route('/summary/<start>/<end>')
def rest_show_summary(start, end):
    # TODO: setup - pagination using start and end
    start = int(start) if start.isdigit() else 0
    end = int(end) if end.isdigit() else 10
    # from random import shuffle
    receipt_images = image_manager.GetNewImage.receipt_images
    data_list = [(key, receipt_images[key]) for key in image_manager.GetNewImage.pending_images]
    data_list = sorted(data_list)[start: end]
    # shuffle(data_list)
    data_dict = dict(data_list)
    return jsonify({'receipt_images': data_dict})


@app.route("/cropper/<image_name>")
def page_cropper(image_name=None):
    data = {'image_name': "/" + os.path.join(config.IMAGE_ROOT_DIR, image_name)}
    return render_template("Cropper_js.html", data=data)


@app.route("/cropper/upload", methods=['POST'])
@app.route("/upload", methods=['POST'])
def cropper_upload():
    data = dict(request.form)
    print('Cropper - File upload status: {}'.format(cropper.cropper_upload(data)))
    return 'ok'


@app.route("/stats")
def stats_page():
    data = stats.get_stats()
    # print(data)
    return render_template("stats.html", html_data=Markup(data))


if __name__ == '__main__':
    app.run(debug=True)  # host='172.16.49.198'
