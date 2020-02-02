import os    
from google_drive_downloader import GoogleDriveDownloader as gdd    

DATA = ['train.csv', 'test.csv']

train_id = '11GZW9Li5nNsyvcYtjWipi0kAuYlmZ0bb'
test_id = '1LpG-zath2UpUFtPB1qd7jDbj5vh0Pbro'

def main(output_dir='data'):
    filenames = DATA
    ids = [train_id, test_id]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for id, filename in zip(ids, filenames):
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            print("{} already exists".format(output_file))
            continue

        print("Downloading from {} ...".format(id))
        gdd.download_file_from_google_drive(file_id=id,
                                    dest_path='./'+output_dir+'/'+filename)
        print("=> File saved as {}".format(output_file))

if __name__ == '__main__':
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()
              
