import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.change_pt_detect_model import ChangePtDetectModel
from trainers.change_pt_detect_trainer import ChangePtDetectTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
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
	create_dirs([config.summary_dir, config.checkpoint_dir])

	# create tensorflow session
	sess = tf.Session()
	# create instance of the model you want
	model = ChangePtDetectModel(config)

	# create your data generator
	data = DataGenerator(config)

	# create tensorboard logger
	logger = Logger(sess, config)

	# create trainer and path all previous components to it
	trainer = ChangePtDetectTrainer(sess, model, data, config, logger)

	# here you train your model
	trainer.train()

	# save trained model
	model.save(sess)


if __name__ == '__main__':
    main()
