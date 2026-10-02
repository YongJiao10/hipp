function run_kong2022_areal_mshbm_subject(project_dir, output_dir, run_list_file, censor_list_file, cbig_code_dir, group_prior, overwrite_flag)
% Run Kong2022_ArealMSHBM for one fsLR32k subject from dtseries sessions.

if nargin < 7
    overwrite_flag = 0;
end

setenv('CBIG_CODE_DIR', cbig_code_dir);
addpath(fullfile(cbig_code_dir, 'stable_projects', 'brain_parcellation', 'Kong2022_ArealMSHBM'));
addpath(genpath(fullfile(cbig_code_dir, 'utilities')));
addpath(genpath(fullfile(cbig_code_dir, 'external_packages')));

run_list = read_nonempty_lines(run_list_file);
censor_list = read_nonempty_lines(censor_list_file);

if length(run_list) ~= length(censor_list)
    error('Run/censor list length mismatch: %d runs vs %d censors', length(run_list), length(censor_list));
end

params = struct();
params.project_dir = project_dir;
params.target_mesh = 'fs_LR_32k';
params.censor_list = censor_list;
params.lh_fMRI_list = run_list;
params.model = 'gMSHBM';
params.num_ROIs = '400';
params.w = '30';
params.c = '30';
params.beta = '50';
params.group_prior = group_prior;
params.overwrite_flag = overwrite_flag;

[lh_labels, rh_labels] = CBIG_ArealMSHBM_parcellation_single_subject(params);

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
writematrix(round(lh_labels(:)), fullfile(output_dir, 'lh_labels.txt'), 'FileType', 'text');
writematrix(round(rh_labels(:)), fullfile(output_dir, 'rh_labels.txt'), 'FileType', 'text');
end

function lines = read_nonempty_lines(path)
fid = fopen(path, 'r');
if fid == -1
    error('Unable to open list file: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
lines = {};
while true
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end
    line = strtrim(line);
    if ~isempty(line)
        lines{end + 1} = line; %#ok<AGROW>
    end
end
if isempty(lines)
    error('List file is empty: %s', path);
end
end
