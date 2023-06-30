clc;
clear;


% real time courses
dataFile = 'X:\home\lihon\comp_space\code_share\LSTM_change_pt_detect\example_data\MOTOR_LR_test.mat';
data = load(dataFile);

% predicted time courses by the LSTM RNNs (output file from the LSTM RNNs model)
resFile = 'X:\home\lihon\comp_space\code_share\LSTM_change_pt_detect\results\hcp_motor_lr_test\output_prediction.mat';
res = load(resFile);

% 
num_data = size(res.y_,1);       % number of subjects
numT = 284;                                % number of time points
task_block = 3:1:7;                     % indices for task blocks

% Identify the change points using selected brain ROIs/functional networks or
% all ROIs (when set to [])
task_ic_id = [1, 5, 8, 11, 14, 30, 32, 57, 66, 67, 74, 80, 86]; %[]; 

% sigma and lambda are hyper-parameters for identifying change points, may need
% adjustments for different datasets
sigma = 6;              % width factor in Gaussian smoothing
lambda = 0;             % control the threshold value used to identify the chagne points

testing_di = 51:num_data;       % index for the testing subjects
lstm_cpt_all = cell(length(testing_di), 1);
for ndi=1:length(testing_di)
    raw_data = squeeze(data.t_y(ndi,:,:));
    reg_data = squeeze(res.y_(ndi,:,:));

    if ~isempty(task_ic_id)
        raw_data = raw_data(:,task_ic_id);
        reg_data = reg_data(:,task_ic_id);
    end
    
    %% identify the change points based on the prediction error
    err = sqrt(sum((raw_data(1:end-1,:)-reg_data(1:end-1,:)).^2,2));
    err = err ./ sqrt(sum(raw_data(1:end-1,:).^2, 2));

    err(1:5) = [];
    s_err = (err-min(err)) ./ (max(err)-min(err));

    sw = gausswin(32, sigma);
    sw = sw ./ sum(sw);
    s_err = conv(s_err, sw, 'same') * 5;

    [epeaks, elocs] = findpeaks(double(s_err));

    mean_err = mean(s_err);
    std_err = std(s_err);
    thr_val = mean_err + std_err * lambda;

    lstm_cpt = elocs(epeaks>=thr_val) + 5;      % change points identified identified by LSTM RNNs
    
    lstm_cpt_all{ndi} = lstm_cpt;
end
disp('Finished.');

