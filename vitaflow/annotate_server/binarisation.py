import subprocess


def binarisation(image_loc, dest_image_loc):
    # print('binarisation src {} dest {} '.format(image_loc, dest_image_loc))
    subprocess.check_call(['./textcleaner',
                           '-g',
                           '-e',
                           'stretch',
                           '-f',
                           '25',
                           '-o',
                           '10',
                           '-u',
                           '-s',
                           '1',
                           '-T',
                           '-p',
                           '10',
                           image_loc, dest_image_loc])


# noinspection PyUnusedLocal
def blur(image_loc, dest_image_loc):
    pass
