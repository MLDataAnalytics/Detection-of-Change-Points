import tensorflow as tf
import numpy as np
import scipy.io as sio
import os

from models.change_pt_detect_model import ChangePtDetectModel
from utils.config import process_config
from utils.dirs import create_dirs
#from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.test_output_dir])

    # create tensorflow session
    sess = tf.Session()

    # create instance of the model you want
    model = ChangePtDetectModel(config)
    # load model
    model.load(sess)

    # get test data
    mat_input = sio.loadmat(config.test_data)

    x = mat_input["t_x"]
    seq_len = mat_input["t_len"].flatten()
    #y = mat_input["t_y"]
    #y_mask = mat_input["t_mask"]

    N_test = x.shape[0]

    output = []
    batch_size = config.batch_size
    num_iter = np.int32(np.ceil(N_test/batch_size))

    for bi in range(num_iter):
        if bi < num_iter-1:
            run_ind = range(bi*batch_size, (bi+1)*batch_size)
        else:
            run_ind = list(range(bi*batch_size, N_test)) + list(range(num_iter*batch_size-N_test))
        
        r_sequences = x[run_ind,:,:]
        r_seq_len = seq_len[run_ind]
        bi_output = sess.run([model.y_], {model.x: r_sequences, model.seq_len: r_seq_len})
        
        bi_y_ = bi_output[0]
        if bi == num_iter-1:
            bi_y_ = bi_output[0][0:N_test-(num_iter-1)*batch_size,:,:]
                
        output.append(bi_y_)

    output = np.concatenate(output, axis=0)

    output_file = os.path.join(config.test_output_dir, r"output_prediction.mat")
    sio.savemat(output_file, {'y_': output})


if __name__ == '__main__':
    main()
