from __future__ import print_function

import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


URLBASE = 'https://ndownloader.figshare.com/files/{}'
URLS = ['18473234', '18475136', '18474977']
DATA = ['AirBnB_train.csv.zip', 'AirBnB_test.csv.zip']

train_url = 'https://www.dropbox.com/s/pftm879cilwq9hz/train.csv?dl=0'
test_url = 'https://www.dropbox.com/s/zt5jrweae1s6y1o/test.csv?dl=0'

def main(output_dir='data'):
    filenames = DATA
    urls = [train_url, test_url]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for url, filename in zip(urls, filenames):
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            print("{} already exists".format(output_file))
            continue

        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))

if __name__ == '__main__':
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()