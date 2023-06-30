
Requirements:
	tensorflow (version <=1.15)
	numpy
	scipy
	tqdm
	bunch


Usage:
1. LSTM training:
	python main_train.py -c example_config.json

	A example config file can be found here: ./configs/changePtDetect_config_Motor.json

2. LSTM testing:
	python main_test.py -c example_config.json

	The prediction results will be saved as 'output_prediction.mat' in the 'test_output_dir' as set in the config.json file

3. LSTM configuration file (.json):
	{
	  "exp_name": "changePtDetect_hcp_motor",		# name for the experiment/model
	  "train_data": "/cbica/home/lihon/comp_space/code_share/LSTM_change_pt_detect/example_data/MOTOR_LR_train.mat",	# location for the training data
	  "is_training": "False",						# Training (True) or testing (False) the model?
	  "model_output": "/cbica/home/lihon/comp_space/code_share/LSTM_change_pt_detect/results/hcp_motor_lr_model",		# where the trained model will be saved
	  "test_data": "/cbica/home/lihon/comp_space/code_share/LSTM_change_pt_detect/example_data/MOTOR_LR_test.mat",		# location for the testing data
	  "test_output_dir": "/cbica/home/lihon/comp_space/code_share/LSTM_change_pt_detect/results/hcp_motor_lr_test",		# where the testing results will be saved
	  "num_epochs": 20,	
	  "num_iter_per_epoch": 10000,
	  "learning_rate": 0.001,
	  "decay_steps": 50000,
	  "decay_rate": 0.1,
	  "batch_size": 8,
	  "step_num": 284,								# number of time points of the fMRI data used for training and testing
	  "fea_num": 90,								# number of brain ROIs/functional networks
	  "num_hidden_layers": 2,						# number of hidden layers in the LSTM RNNs
	  "hidden_num": 256,							# number of hidden nodes in each LSTM layer
	  "dropout_keep_rate": 1.0,
	  "max_to_keep": 5
	}

4. Prepare the training/testing data:
	Currently the training and testing data files are saved in .mat format (Matlab file).
	The data file contains:
		t_x: 3D tensor with size [num_sbj, num_time_points, num_roi], the input time courses (for time points 0, 1, ..., T-1) to the LSTM RNNs
		t_y: 3D tensor with the same size as t_x, the time courses (for time points 1, 2, ..., T) used as the groundtruth output 
		t_len: array with size [num_sbj, 1], containing the number of time points of the time courses for each subject
		t_mask: 3D tensor with the same size as t_x, only the time points and ROIs within the mask will be used to compute the loss during training

	An example Matlab script for preparing the data can be found here: ./m_scripts/prep_data_for_tf_lstm.m
	An example data file can be found here: ./example_data/MOTOR_LR_test.mat

5. Change point detection from the LSTM output
	The change points can be identified based on the output file from the LSTM model and the real time courses, using the Matlab script here: ./m_scripts/get_change_pts_lstm.m
