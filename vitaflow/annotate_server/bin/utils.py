import glob
import os


def trim_file_ext(x):
    if '.' in x:
        return x.rsplit('.')[-2]
    else:
        return x


def check_n_create(full_path):
    if not os.path.isdir(full_path):
        print('Created missing dir `{}`!'.format(full_path))
        os.mkdir(full_path)
    else:
        print('Using dir at `{}`!'.format(full_path))


def parser_folder(search_folder, exts=None):
    "Return (base_filename, full_path_filename)"
    bag = []
    for filename in glob.iglob(search_folder + '/*', recursive=True):
        file = os.path.basename(filename)
        if exts:
            for _ext in exts:
                if file.endswith(_ext):
                    bag.append((file, filename))
                    break
        else:
            bag.append((file, filename))
    return bag


def get_folder_config(dir_path, file_exts, trim_path_prefix):
    file_dict = {}
    _found_files = parser_folder(dir_path, file_exts)
    for file, full_path_filename in _found_files:
        url = full_path_filename.split(trim_path_prefix)[-1].lstrip(os.sep)
        file = os.path.basename(url)
        # print(url, file)
        file_dict[trim_file_ext(file)] = {
            'url': url,
            'file': file
        }
    return file_dict
