function run_kong2022_mshbm_subject(project_dir, cbig_code_dir, run_list_file, group_prior, model, num_rois, w, c, beta, censor_list_file)
% run_kong2022_mshbm_subject
% Thin Kong2022_ArealMSHBM single-subject wrapper for fs_LR_32k dtseries runs.

if nargin < 9
    beta = '50';
end
if nargin < 10
    censor_list_file = '';
end
if nargin < 8
    c = '30';
end
if nargin < 7
    w = '30';
end
if nargin < 6
    num_rois = '400';
end
if nargin < 5
    model = 'gMSHBM';
end

if isnumeric(num_rois); num_rois = num2str(num_rois); end
if isnumeric(w); w = num2str(w); end
if isnumeric(c); c = num2str(c); end
if isnumeric(beta); beta = num2str(beta); end

if ~exist(project_dir, 'dir')
    mkdir(project_dir);
end

setenv('CBIG_CODE_DIR', cbig_code_dir);
kong2022_dir = fullfile(cbig_code_dir, 'stable_projects', 'brain_parcellation', 'Kong2022_ArealMSHBM');
if ~exist(kong2022_dir, 'dir')
    error('Missing Kong2022_ArealMSHBM directory: %s', kong2022_dir);
end

addpath(kong2022_dir);
addpath(genpath(fullfile(kong2022_dir, 'step0_generate_gradient_prior')));
addpath(genpath(fullfile(kong2022_dir, 'step1_generate_profiles_and_ini_params')));
addpath(genpath(fullfile(kong2022_dir, 'step2_estimate_priors')));
addpath(genpath(fullfile(kong2022_dir, 'step3_generate_ind_parcellations')));

util_dir = fullfile(cbig_code_dir, 'utilities');
if exist(util_dir, 'dir')
    addpath(genpath(util_dir));
end
external_pkg_dir = fullfile(cbig_code_dir, 'external_packages');
if exist(external_pkg_dir, 'dir')
    addpath(genpath(external_pkg_dir));
end

runs = read_text_lines(run_list_file);
if isempty(runs)
    error('No runs found in run_list_file: %s', run_list_file);
end
censors = read_text_lines(censor_list_file);
if isempty(censors)
    error('No censor rows found in censor_list_file: %s', censor_list_file);
end
if numel(censors) ~= numel(runs)
    error('censor list count (%d) does not match run count (%d)', numel(censors), numel(runs));
end

params = struct();
params.project_dir = project_dir;
params.target_mesh = 'fs_LR_32k';
params.lh_fMRI_list = runs;
params.censor_list = censors;
params.model = model;
params.num_ROIs = num_rois;
params.w = w;
params.c = c;
params.beta = beta;
params.group_prior = group_prior;
params.overwrite_flag = 1;

[lh_labels, rh_labels] = CBIG_ArealMSHBM_parcellation_single_subject(params);

out_dir = fullfile(project_dir, 'route_a_outputs');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

write_vector_txt(fullfile(out_dir, 'lh_labels.txt'), lh_labels);
write_vector_txt(fullfile(out_dir, 'rh_labels.txt'), rh_labels);
save(fullfile(out_dir, 'route_a_outputs.mat'), ...
    'lh_labels', 'rh_labels', 'model', 'num_rois', 'w', 'c', 'beta', 'group_prior', 'runs');

fprintf('Kong2022 ArealMSHBM wrapper done. project_dir=%s\n', project_dir);
end


function lines = read_text_lines(path)
fid = fopen(path, 'r');
if fid == -1
    error('Failed to open run list file: %s', path);
end
lines = {};
while true
    row = fgetl(fid);
    if ~ischar(row)
        break;
    end
    row = strtrim(row);
    if isempty(row)
        continue;
    end
    lines{end+1,1} = row; %#ok<AGROW>
end
fclose(fid);
end


function write_vector_txt(path, vec)
fid = fopen(path, 'w');
if fid == -1
    error('Failed to open output label file: %s', path);
end
for idx = 1:numel(vec)
    fprintf(fid, '%.0f\n', vec(idx));
end
fclose(fid);
end
