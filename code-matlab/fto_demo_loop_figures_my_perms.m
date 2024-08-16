% fto_demo

% changed corepath and function path (PJH)
% corepath = '/Volumes/fto2/';
corepath = '/home/phebden/Glasgow/DRP-3-code';
addpath([corepath,'/GAR/functions'])

% Note about plots: if figure NumberTitle is 'off', change to show titles

% this script will save output to this folder (PJH)
data_folder = 'ftonsets_results';
plot_folder = 'ftonsets_plots';

% the ERP data for the ftonsets project are available here:
% https://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.46786

% LIMO EEG is available here:
% https://github.com/LIMO-EEG-Toolbox/limo_tools

%% LOAD DATA

fname_onsets = sprintf('%s/ftonsets_demographics/ftonsets_2ses', corepath);
load(fname_onsets); 

ath = 0.05;    % arbitrary threshold value for ttest
pth = 1 - ath; % permutation threshold

rng(21) % set random seed

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% set sim parameters, do a few participants to test, example plots, etc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nperm = 2; % 100 number of permutations -- use 1000 in non-demo scripts

% generate plots for some or no participants
plot_p_number = 1;  % 0 for none, 1 for participant 1, or e.g. first 5

participant_numbers = 1:1;  % 120;  % <<<<=============-------------------
session_numbers     = 1:1;

session_count = 0;

close_all_figs = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_participants=120;
participants_s1_t2 = cell(1, num_participants);
participants_s2_t2 = cell(1, num_participants);

participants_s1_maxt2 = cell(1, num_participants);
participants_s2_maxt2 = cell(1, num_participants);

% 120 participants, 74 did 2 sessions, NA if no session 2
participants_s1_onsets=zeros(1, num_participants);
participants_s2_onsets=zeros(1, num_participants);


%%
tStart = tic;   % start the timer

for p_number = participant_numbers

    for s_number = session_numbers

    if s_number == 2 && ftonsets_2ses(p_number) == 0

        % no session 2 for this participant
        participants_s2_t2{p_number}     = NaN; % just using 1 NaN instead of a matrix
        participants_s2_maxt2{p_number}  = NaN; 
        participants_s2_onsets(p_number) = NaN; 
        continue                      % only 1 session so do next participant
    end

    session_count = session_count + 1;

    %fprintf("participant %d, session %d, permutations %d\n", p_number, s_number, Nperm); % PJH

    % load the data struct
    fname_data = sprintf('%s/ftonsets_erps_avg/ftonsets_p%d_s%d',corepath, p_number, s_number);
    load(fname_data); 

    ferp = data.face_trials;  % 3D matrix electrodes x time points x trials
    nerp = data.noise_trials; % 3D matrix electrodes x time points x trials
    [Ne,Nf,Ntf] = size(ferp);
    Ntn = size(nerp,3);
    Xf = data.times;
    
    maxt2_perm = zeros(Nf,Nperm);
%% plot erp for 1 electrode

electrode_num = 1;

% RED = FACE ERPs
% GREY = NOISE ERPs
% note the large N170 between 100-200 ms.
if p_number <= plot_p_number

    figname = sprintf('ERP for electrode number 1: p%d s%d', p_number, s_number);
    figure('Name', figname,'NumberTitle','on','Color','white');    
    temp = mean(ferp,3)';       % 451 time pts x 119 electrodes (if participant 1)
    temp = temp(:,electrode_num);
    face_line = plot(Xf, temp, 'Color', [1 0 0], 'LineWidth',3);    % face ERP = mean across 3rd dimension
    hold on;

    temp = mean(nerp,3)';
    temp = temp(:,electrode_num);
    noise_line = plot(Xf, temp, 'Color', [.7 .7 .7] , 'LineWidth',3); % noise ERP = mean across 3rd dimension
    hold on;
    title(figname)    
    set(gca, 'Layer', 'top')
    xlabel('time (ms)')
    ylabel('amplitude (\muV)')
    ylim([-25,25]);
    ax = gca;
    ax.FontSize = 14;
    legend([face_line, noise_line], {'face', 'texture'});
%  FIG
fname=sprintf("%s/p%d_s%d_erps_electrode_%d.png", plot_folder, p_number, s_number, electrode_num);
saveas(gcf,fname)
end

%% PLOT ERPS AT ALL ELECTRODES

% RED = FACE ERPs
% GREY = NOISE ERPs
% note the large N170 between 100-200 ms.
if p_number <= plot_p_number

    figname = sprintf('ERPs per electrode: p%d s%d', p_number, s_number);
    figure('Name', figname,'NumberTitle','on','Color','white');    
    for temp = mean(ferp,3)'
        face_line = plot(Xf, temp, 'Color', [1 0 0] );    % face ERPs = mean across 3rd dimension
        hold on;
    end
    for temp = mean(nerp,3)'
        noise_line = plot(Xf, temp, 'Color', [.7 .7 .7]); % noise ERPs = mean across 3rd dimension
        hold on;
    end
    title(figname)    
    set(gca, 'Layer', 'top')
    xlabel('time (ms)')
    ylabel('amplitude (\muV)');
    ylim([-25, 25])
    ax = gca;
    ax.FontSize = 14;
    legend([face_line, noise_line], {'face', 'texture'});
%  FIG
fname=sprintf("%s/p%d_s%d_erps.png", plot_folder, p_number, s_number);
saveas(gcf,fname)
end
%% PLOT face ERPS AT ALL ELECTRODES on separate line

% RED = FACE ERPs
% GREY = NOISE ERPs
% note the large N170 between 100-200 ms.
% if p_number <= plot_p_number
% figname = sprintf('Average face ERPs per electrode: p%d s%d', p_number, s_number);
% fig =figure('Name', figname, 'NumberTitle', 'on', 'Color', 'white');
% fig.Position = [100 100 640 1600];
% 
% % Determine the number of lines to plot
% num_face_lines = 30; %size(ferp, 1);
% %num_noise_lines = size(nerp, 1);
% 
% % Plot face ERPs in separate rows
% for i = 1:num_face_lines
% 
%     mean_erp = mean(ferp(i, :, :), 3);
%     mean_erp = mean_erp + (i * 45);
%     plot(Xf, mean_erp, 'Color', [0 0 .8], 'LineWidth', 2); % Face ERPs
%     hold on;
%     % mean_erp = mean(nerp(i, :, :), 3);
%     % mean_erp = mean_erp + (i * 20);
%     % plot(Xf, mean_erp, 'Color', [.7 .7 .7], 'LineWidth',2); % Face ERPs
% end
% 
% %title(sprintf('Face ERP %d', i));
% %ylim([-25, 25]);
% set(gca,'YTick',[])
% ylabel('amplitude (\muV)');
% xlabel('time (ms)');


% Plot noise ERPs in separate rows
% for j = 1:num_noise_lines
%     subplot(num_face_lines + num_noise_lines, 1, num_face_lines + j);
%     plot(Xf, mean(nerp(j, :, :), 3), 'Color', [.7 .7 .7]); % Noise ERPs
%     hold on;
%     title(sprintf('Noise ERP %d', j));
%     ylim([-25, 25]);
%     ylabel('mean ERPs (\muV)');
%     if j == num_noise_lines
%         xlabel('time (ms)');
%     end
% end
% 
% sgtitle(figname); % Super title for the whole figure
% set(gca, 'Layer', 'top');
% ax = gca;
% ax.FontSize = 14;
% %legend({'face', 'noise'});
% 
% 
% %  FIG
% fname=sprintf("%s/p%d_s%d_face_erps_rows.png", plot_folder, p_number, s_number);
% saveas(gcf,fname)
% end
%% noise ERPs

% if p_number <= plot_p_number
% figname = sprintf('Average texture ERPs per electrode: p%d s%d', p_number, s_number);
% fig = figure('Name', figname, 'NumberTitle', 'on', 'Color', 'white');
% fig.Position = [100 100 640 1600];
% 
% % Determine the number of lines to plot
% num_face_lines = 30; %size(ferp, 1);
% %num_noise_lines = size(nerp, 1);
% 
% % Plot face ERPs in separate rows
% for i = 1:num_face_lines
% 
%     mean_erp = mean(nerp(i, :, :), 3);
%     mean_erp = mean_erp + (i * 45);
%     plot(Xf, mean_erp, 'Color', [.5 .5 .5], 'LineWidth', 2); % Face ERPs
%     hold on;
%     % mean_erp = mean(nerp(i, :, :), 3);
%     % mean_erp = mean_erp + (i * 20);
%     % plot(Xf, mean_erp, 'Color', [.7 .7 .7], 'LineWidth',2); % Face ERPs
% end
% 
% %title(sprintf('Face ERP %d', i));
% %ylim([-25, 25]);
% set(gca,'YTick',[])
% ylabel('amplitude (\muV)');
% xlabel('time (ms)');
% 
% sgtitle(figname); % Super title for the whole figure
% set(gca, 'Layer', 'top');
% ax = gca;
% ax.FontSize = 14;
% %legend({'noise'});
% 
% fname=sprintf("%s/p%d_s%d_noise_erps_rows.png", plot_folder, p_number, s_number);
% saveas(gcf,fname)
% end

%% COMPUTE ORIGINAL T-TESTS 

% t-tests
[~, ~, ~, ~, ~, tval, ~] = limo_ttest(2,ferp,nerp,ath);
maxt2 = max(tval.^2, [], 1);

% assign tvals to matrix
if s_number == 1
    participants_s1_t2{p_number} = tval.^2;
elseif s_number == 2 
    participants_s2_t2{p_number} = tval.^2;
end


%% PLOT T VALUES

% if p_number <= plot_p_number
%     figname = sprintf('t-values: p%d s%d', p_number, s_number);
%     figure('Name', figname,'NumberTitle','on','Color','white');
%     plot(Xf, tval, 'Color', [0 0 0]) 
%     title(figname)
%     set(gca, 'Layer', 'top')
%     xlabel('time (ms)')
%     ylabel('t-values')
%     ax = gca;
%     ax.FontSize = 14;
% 
% fname=sprintf("%s/p%d_s%d_tval.png", plot_folder, p_number, s_number);
% saveas(gcf,fname)
% end

%% PLOT t^2 VALUES
if p_number <= plot_p_number
    figname = sprintf('t^2 values: p%d s%d', p_number, s_number);
    figure('Name', figname,'NumberTitle','on','Color','white');
    plot(Xf, tval.^2, 'Color', [0 0 0]) 
    title(figname)
    set(gca, 'Layer', 'top')
    xlabel('time (ms)')
    ylabel('t^2 values')
    ax = gca;
    ax.FontSize = 14;
fname=sprintf("%s/p%d_s%d_t2.png", plot_folder, p_number, s_number);
saveas(gcf,fname)

end

%% load virtual electrode's max t2 for session 1 and session 2

% this is session levels

% if s_number == 1
%     maxt2 = load('max_t2_avg_all_s1.csv');
% end 
% 
% if s_number == 2
%     maxt2 = load('max_t2_avg_all_s2.csv');
% end 

%% PLOT T^2 MAX VALUES

% if p_number <= plot_p_number
%     figname = sprintf('max t^2: p%d s%d', p_number, s_number);
%     figure('Name', figname,'NumberTitle','on','Color','white');
%     plot(Xf, maxt2, 'Color', [1 0 0],'LineWidth', 2)
%     hold on 
% 
%     %plot(Xf, tval.^2, 'Color', [0.7 0.7 0.7]) 
% 
%     title(figname)
%     set(gca, 'Layer', 'top')
%     xlabel('time (ms)')
%     ylabel('t^2 values')
%     ax = gca;
%     ax.FontSize = 14;
%     %legend('max t^2', 't^2')
%     legend('max t^2')
% 
% % fname=sprintf("%s/p%d_s%d_t2_max.png", plot_folder, p_number, s_number);
% % saveas(gcf,fname)
% end

%% COMPUTE PERMUTATION T-TESTS

% create random permutation samples
% use same samples for ttest and MI
indx_perm = zeros(Nperm, Ntf + Ntn);
for perm_iter = 1:Nperm % same trials used for all electrodes and time points
    indx_perm(perm_iter,:) = randperm(Ntf + Ntn);
end

% get permutation estimates
erpall = cat(3,ferp,nerp);
prctval_perm = zeros(Ne,Nf,Nperm); % bivariate: EEG + gradient
for perm_iter = 1:Nperm
    perm_trials = indx_perm(perm_iter,:);
    perm_ferp = erpall(:,:,perm_trials(1:Ntf));
    perm_nerp = erpall(:,:,perm_trials(Ntf+1:Ntf+Ntn));
    [~, ~, ~, ~, ~, tval, ~] = limo_ttest(2,perm_ferp,perm_nerp,ath);
    maxt2_perm(:,perm_iter) = max(tval.^2, [], 1);
end

%% PLOT PERMUTATION ESTIMATES OF MAX T^2 VALUES

if p_number <= plot_p_number
    figname = sprintf('max t^2: p%d s%d perms %d', p_number, s_number, Nperm);
    figure('Name', figname,'NumberTitle','on','Color','white');
    plot(Xf, maxt2, 'r', 'LineWidth', 2)
    hold on  
    plot(Xf, maxt2_perm, 'Color', [.7 .7 .7])
    title(figname)
    set(gca, 'Layer', 'top')
    xlabel('time (ms)')
    ylabel('t^2 values')
    ax = gca;
    ax.FontSize = 14;
    legend("max t^2", "max t^2 perm");   

fname=sprintf("%s/p%d_s%d_permutations.png", plot_folder, p_number, s_number);
saveas(gcf,fname)

end

%% CLUSTER BASED STATISTICS

% baseline normalisation -- subtract 50th quantile of baseline T2
% changed hd (see functions folder) to Matlab's median, as suggested by GAR
for perm_iter = 1:Nperm
    temp = maxt2_perm(:,perm_iter) - median(maxt2_perm(Xf<0,perm_iter));
    temp(temp < 0) = 0;
    maxt2_perm(:,perm_iter) = temp; % maxt2_perm(:,perm_iter) - median(maxt2_perm(Xf<0,perm_iter));
end
maxt2 = maxt2 - prctile(maxt2(Xf<0), 50);   % but prctile(x,50) == median(x) == hd(x)

if size(maxt2,1) == 1
    maxt2 = maxt2';
end

maxt2(maxt2 < 0)=0;

% store the maxt2 vector in a cell
if s_number == 1
    participants_s1_maxt2{p_number}=maxt2;
elseif s_number == 2
    participants_s2_maxt2{p_number}=maxt2; 
end
%%

if false==true
    temp_perm = maxt2_perm(:);  % flatten
     
    % Calculate the 95th percentile
    pct_95 = prctile(temp_perm, 95);
    
    figname = sprintf('Max t^2 permutations: p%d s%d', p_number, s_number);
    figure('Name', figname,'NumberTitle','on','Color','white');
    %h1=plot(Xf, maxt2_perm(:, 2), 'r', 'LineWidth', 2);     % 451 time pts x n_perms
    h1=histogram(temp_perm,20); % 'BinEdges', 0:1:14); % Specify bins from 0 to 14 with bin width of 1
    hold on;
    plot([pct_95 pct_95], [0 max(h1.BinCounts)],  'r', 'LineStyle', ':', 'LineWidth', 2)
    xlabel('t{^2} value')
    ylabel('count')
    
    ax = gca;
    ax.YAxis.Exponent = 0;
    ax.FontSize = 14;
    legend('max t{^2}', '95th percentile')
    title(figname)
    fname=sprintf("%s/p%d_s%d_max_t2_perm_distr.png", plot_folder, p_number, s_number);
    saveas(gcf,fname)
    
    figname = sprintf('Max t{^2} values: p%d s%d', p_number, s_number);
    figure('Name', figname,'NumberTitle','on','Color','white');
    %h1=plot(Xf, maxt2_perm(:, 2), 'r', 'LineWidth', 2);     % 451 time pts x
    %Nperm
    
    h2=histogram(maxt2, 20); %, 'BinEdges', 0:1:14); 
    hold on;
    plot([pct_95 pct_95], [0 max(h2.BinCounts)], 'r', 'LineStyle', ':', 'LineWidth', 2)
    xlabel('t{^2} value')
    ylabel('count')
    ax = gca;
    ax.FontSize = 14;
    legend('max t{^2}', 'threshold')
    title(figname)
    fname=sprintf("%s/p%d_s%d_max_t2_distr.png", plot_folder, p_number, s_number);
    saveas(gcf,fname)
end

%%

% Initialize clusters
clusters = {};
current_cluster = [];
cluster_sum_vec=[];

% permuation vector with many values between 0 and max t2

dims=size(maxt2_perm);

for col_num = 1:dims(2)
    data =  maxt2_perm(:,col_num);

    % Find indices of values >= than threshold
    indices = find(data >= pct_95);
    current_cluster = [];

    % Loop through indices to cluster contiguous values
    for i = 1:length(indices)
        if isempty(current_cluster)
            current_cluster = indices(i);
        elseif indices(i) == indices(i-1) + 1
            current_cluster = [current_cluster, indices(i)];
        else
            clusters{end+1} = current_cluster;
            current_cluster = indices(i);
        end
    end
    
    % Add the last cluster if it exists
    if ~isempty(current_cluster)
        clusters{end+1} = current_cluster;
    end
end

% Display the clusters and their values
for i = 1:length(clusters)
    cluster_indices = clusters{i};
    %cluster_onset(i) = clusters{i}(1);
    cluster_values = data(cluster_indices);
    cluster_sum=sum(cluster_values);
    cluster_sum_vec = [ cluster_sum_vec, [sum(cluster_values)] ];
    %fprintf('Cluster %d: Indices = %s, Values = %s\n', i, cluster_sum, mat2str(cluster_indices), mat2str(cluster_values));
end

% Calculate the 95th percentile
cs_pct_95 = prctile(cluster_sum_vec, 95);
% Find indices of values >= than cs_pct_95
indices = find(cluster_sum_vec >= cs_pct_95);
idx=indices(1);
cs_threshold=cluster_sum_vec(idx);
fprintf("cluster sum threshold = %0.3f", cs_threshold);
%%

% max_t2 vector with 451 values between 0 and the max t2
data = maxt2;

% Find indices of values >= than threshold
indices = find(data >= pct_95);

% Initialize clusters
clusters = {};
current_cluster = [];
cluster_sum_vec_maxt2=[];

% Loop through indices to cluster contiguous values
for i = 1:length(indices)
    if isempty(current_cluster)
        current_cluster = indices(i);
    elseif indices(i) == indices(i-1) + 1
        current_cluster = [current_cluster, indices(i)];
    else
        clusters{end+1} = current_cluster;
        current_cluster = indices(i);
    end
end

% Add the last cluster if it exists
if ~isempty(current_cluster)
    clusters{end+1} = current_cluster;
end

cluster_onsets=[];

% Display the clusters and their values
for i = 1:length(clusters)
    cluster_indices= clusters{i};
    cluster_values = data(cluster_indices);
    cluster_sum=sum(cluster_values);
    cluster_sum_vec_maxt2 = [ cluster_sum_vec_maxt2, [sum(cluster_values)] ];
    fprintf('Cluster %d: sum = %0.3f\n', i, cluster_sum);
end

% Calculate the 95th percentile
% cs_pct_95 = prctile(cluster_sum_vec_max_t2, 95);
% % Find indices of values >= than cs_pct_95
indices = find(cluster_sum_vec_maxt2 >= cs_pct_95);
idx=indices(1);
fprintf("cluster sum = %0.3f\n", cluster_sum_vec(idx));

%%

figname = sprintf('Cluster-sum max t^2 permutations, p%d s%d', p_number, s_number);
figure('Name', figname, 'NumberTitle','on','Color','white');
%h1=plot(Xf, maxt2_perm(:, 2), 'r', 'LineWidth', 2);     % 451 time pts x n_perms
h1=histogram(cluster_sum_vec); 
hold on;
plot([cs_pct_95 cs_pct_95], [0 max(h1.BinCounts)], 'r', 'LineStyle', ":", 'LineWidth', 2)
xlabel('t{^2} cluster-sum')
ylabel('count')
title(figname)
ax = gca;
ax.FontSize = 14;
legend('cluster-sum', '95th percentile')
fname=sprintf("%s/p%d_s%d_max_t2_perm_CS_distr.png", plot_folder, p_number, s_number);
saveas(gcf,fname)

figname = sprintf('Cluster-sum max t^2 values, p%d s%d', p_number, s_number);
figure('Name', figname,'NumberTitle','on','Color','white');
h1=histogram(cluster_sum_vec_maxt2, 1000, EdgeColor='b'); 
hold on;
plot([cs_pct_95 cs_pct_95], [0 max(h1.BinCounts)], 'r', 'LineStyle', ":", 'LineWidth', 2)
xlabel('t{^2} cluster-sum')
ylabel('count')
ax = gca;
ax.FontSize = 14;
legend('cluster-sum', 'threshold')
title(figname)
fname=sprintf("%s/p%d_s%d_max_t2_CS_distr.png", plot_folder, p_number, s_number);
saveas(gcf,fname)

%%

% CLUSTER ONSETS ==============================
% get threshold
perm_th = prctile(maxt2_perm, pth*100, 2); % univariate thresholds
% because there is no p value for max t^2,
% we use the univariate permutation distributions to threshold themselves
% fake p values: 0 if above threshold, 1 otherwise
pval = maxt2_perm < repmat(perm_th, 1, Nperm);
% threshold permutation distribution
tmp = maxt2_perm;
th  = limo_ecluster_make(tmp, pval, ath);
% threshold T2 results
sigcluster = limo_ecluster_test(maxt2, maxt2<perm_th, th, ath);
% find onset
onset = find_onset(sigcluster.elec_mask, Xf, 1);

fprintf("using p%d s%d, onset %dms \n", p_number, s_number, onset(1))

if s_number == 1
    participants_s1_onsets(p_number)=onset;
elseif s_number == 2
    participants_s2_onsets(p_number)=onset; 
end


%% ILLUSTRATE ONSET RESULTS

if p_number <= plot_p_number
    onset = round(onset); % want integer
    figname = sprintf('Max t^2 cluster onset {%d}ms: p%d s%d', onset, p_number, s_number);
    figure('Name', figname,'NumberTitle','on','Color','white');
    h1=plot(Xf, maxt2, 'r', 'LineWidth', 2);
    hold on
    title(figname)
    for temp = maxt2_perm
        h2 = plot(Xf, temp, 'Color', [.7 .7 .7]);
        hold on
    end 
    plot(Xf, zeros(Nf,1), 'k', 'LineWidth', 1) % zero reference line
    plot(Xf, perm_th, 'k', 'LineWidth', 2)
    v = axis;
    for temp = (sigcluster.elec_mask - 1)
        h3 = plot(Xf, 2000.*temp, 'go', 'MarkerSize', 4);
        hold on;
    end
    plot([onset onset], [v(3) v(4)], 'k:', 'LineWidth', 2)
    plot([0 0], [v(3) v(4)], 'k', 'LineWidth', 1)
    set(gca, 'Layer', 'top')
    ylim([v(3) v(4)])
    xlabel('time (ms)')
    ylabel('t^2')
    ax = gca;
    ax.FontSize = 14;
    legend([h1,h2,h3], {'max t^2', 'max t^2 perms', 'clusters'}, 'Location', 'northeast' )

    fname=sprintf("%s/p%d_s%d_permutations.png", plot_folder, p_number, s_number);
    saveas(gcf,fname)
    end
    
    if close_all_figs == 1
        close all;  
    end

    end   % done session, ready for next participant
 
   if mod(p_number, 5) == 0
     elapsed_time = toc(tStart);
     time_minutes = (elapsed_time/session_count)/60;
     fprintf("done p_number %d, avg time per session %0.2f minutes\n", p_number, time_minutes)
     remaining_sessions = 194-session_count;
     remaining_time = remaining_sessions * time_minutes;
     hours_left = remaining_time/60;
     fprintf("estimated time required for %d sessions: %0.2f hours\n\n", remaining_sessions, hours_left)
   end
end




