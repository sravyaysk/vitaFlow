import os
import tqdm
import time


images_path = '/Users/sampathm/devbox/vitaFlow/vitaFlow/annotate_server/static/data/images/'


def rename_images(path):
    for file in tqdm.tqdm(os.listdir (images_path), desc='Rename Files'):
        _rename_file(images_path, file)


def _rename_file(path, filename):
    if filename.startswith('.'):
        return
    newname = _get_file_newname(filename)
    _old_file = os.path.join(images_path, filename)
    _new_file = os.path.join(images_path, newname)
    # print('Rename file \n\t{} to \n\t{}'.format(_old_file, _new_file))
    os.rename(
        _old_file,
        _new_file
    )


def _get_file_newname(filename):
    base_filename, file_ext = filename.rsplit('.', 1)
    base_filename = base_filename.lower().strip().replace(' ', '_')
    base_filename = ''.join([_ for _ in base_filename if _ in '_1234567890qwertyuiopasdfghjklzxcvbnm'])
    new_filename = '{}.{}'.format(base_filename, file_ext)
    return new_filename


