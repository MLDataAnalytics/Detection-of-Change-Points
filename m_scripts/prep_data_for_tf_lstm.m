clc;
clear;

rt_dir = 'C:\Users\LiHon\Desktop\lstm_change_pt_detect';

task_id = 'MOTOR';
lr_id = 'LR';

input_file = [rt_dir, filesep, 'tc_data', filesep, task_id, '_', lr_id, '_tc.mat'];
% The input file contains:
%       sbjTC: a cell structure with size [num_sbj, 1], each cell contains the
%       time course data for one subject: 2d array, with size [num_time_point, num_roi]
%       sbjId: a cell structure with size [num_sbj, 1], each cell contains the
%       ID for each subject
load(input_file);

fix_len = 10;

t_x = [];
t_y = [];
t_mask = [];
for ni=1:length(sbjTc)
    ni_x = sbjTc{ni};
    
    % data normalization
    m_fix = mean(ni_x(1:fix_len,:));
    ni_x = ni_x ./ max(eps,repmat(m_fix, size(ni_x,1), 1));
    
    m_nt_x = mean(ni_x);
    s_nt_x = std(ni_x);
    ni_x = (ni_x-repmat(m_nt_x, size(ni_x,1), 1)) ./ max(eps, repmat(s_nt_x, size(ni_x,1), 1));
    
    if any(isnan(ni_x(:)))
        error('nan exists.');
    end
    
    t_x = cat(3, t_x, reshape(ni_x,[size(ni_x),1]));
    
    ni_y = zeros(size(ni_x),'single');
    ni_ymask = ni_y;
    
    ni_y(1:size(ni_y,1)-1,:) = ni_x(2:size(ni_x,1),:);
    ni_ymask(1:size(ni_y,1)-1,:) = 1;
    
    t_y = cat(3, t_y, reshape(ni_y,[size(ni_y),1]));
    t_mask = cat(3, t_mask, reshape(ni_ymask,[size(ni_ymask),1]));
end

t_x_tot = permute(t_x, [3,1,2]);
t_y_tot = permute(t_y, [3,1,2]);
t_mask_tot = permute(t_mask, [3,1,2]);

t_len_tot = zeros(size(t_x_tot,1),1,'single');
for ni=1:size(t_x_tot,1)
    t_len_tot(ni) = size(t_x_tot, 2);
end

data_id = {'train', 'test'};
data_ind = {1:400, 401:length(sbjTc)};
for di=1:length(data_id)
    t_x = t_x_tot(data_ind{di},:,:);
    t_y = t_y_tot(data_ind{di},:,:);
    t_mask = t_mask_tot(data_ind{di},:,:);
    t_len = t_len_tot(data_ind{di});
    t_sid = sbjId(data_ind{di});
    
    out_file = [rt_dir, filesep, 'tc_data_tf', filesep, task_id, '_', lr_id, '_', data_id{di}, '.mat'];
    save(out_file, 't_x', 't_y', 't_mask', 't_len', 't_sid');           % The saved model will be used for traing/testing the LSTM RNNs model
end

disp('Finished.');
