import os
import logging

def set_logger(data_name, save_path):
    directory = save_path + '/' + data_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_file = os.path.join(directory, 'train.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
