% fto_demo

% virtual electrode 1 (VE1): for each participant

% change paths to your locations
corepath = '/path-to-parent-folder';
addpath([corepath,'/this-folder/functions'])

% Note about plots: if figure NumberTitle is 'off', change to show titles

% this script will save output to this folder (PJH)
data_folder = 'ftonsets_results';
plot_folder = 'ftonsets_plots';

% the ERP data for the ftonsets project are available here:
% https://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.46786

% LIMO EEG is available here:
% https://github.com/LIMO-EEG-Toolbox/limo_tools

%% LOAD DATA
% data folders are assumed to be in parent folder
fname_onsets = sprintf('%s/ftonsets_demographics/ftonsets_2ses', corepath);
load(fname_onsets); 

ath = 0.05;    % arbitrary threshold value for ttest
pth = 1 - ath; % permutation threshold

rng(21) % set random seed

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% set sim parameters, do a few participants to test, example plots, etc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nperm = 1000; % 100 number of permutations -- use 1000 in non-demo scripts

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

    figname = sprintf('Average ERPs for electrode number 1: p%d s%d', p_number, s_number);
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

    figname = sprintf('Average ERPs per electrode: p%d s%d', p_number, s_number);
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

if p_number <= -1 % plot_p_number
    figname = sprintf('t-values: p%d s%d', p_number, s_number);
    figure('Name', figname,'NumberTitle','on','Color','white');
    plot(Xf, tval, 'Color', [0 0 0]) 
    title(figname)
    set(gca, 'Layer', 'top')
    xlabel('time (ms)')
    ylabel('t-values')
    ax = gca;
    ax.FontSize = 14;

fname=sprintf("%s/p%d_s%d_tval.png", plot_folder, p_number, s_number);
saveas(gcf,fname)
end

%% PLOT t^2 VALUES
if p_number <= -1 % plot_p_number
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

%% PLOT T^2 MAX VALUES

if p_number <= plot_p_number
    figname = sprintf('max t^2 and t^2 values: p%d s%d', p_number, s_number);
    figure('Name', figname,'NumberTitle','on','Color','white');

    plot(Xf, maxt2, 'Color', [1 0 0],'LineWidth', 2)
    hold on
    plot(Xf, tval.^2, 'Color', [0.7 0.7 0.7]) 
    hold on
    plot(Xf, maxt2, 'Color', [1 0 0],'LineWidth', 2)
    legend('max t^2', 't^2 values')

    title(figname)
    %set(gca, 'Layer', 'top')
    xlabel('time (ms)')
    ylabel('t^2 values')
    ax = gca;
    ax.FontSize = 14;


    fname=sprintf("%s/p%d_s%d_max_t2_and_t2_vals.png", plot_folder, p_number, s_number);
    saveas(gcf,fname)
end

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
    [~, ~, ~, ~, ~, tval_perm, ~] = limo_ttest(2,perm_ferp,perm_nerp,ath);
    maxt2_perm(:,perm_iter) = max(tval_perm.^2, [], 1);
end

%% PLOT PERMUTATION ESTIMATES OF MAX T^2 VALUES

if p_number <= plot_p_number
    figname = sprintf('max t^2: p%d s%d, permutations %d', p_number, s_number, Nperm);
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

    fname=sprintf("%s/p%d_s%d_permutations_VE1.png", plot_folder, p_number, s_number);
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
    figname = sprintf('max t^2 cluster onset {%d}ms: p%d s%d', onset, p_number, s_number);
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

    fname=sprintf("%s/p%d_s%d_permutations_VE1_clusters.png", plot_folder, p_number, s_number);
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

%%

elapsed_time = toc(tStart);
time_minutes = (elapsed_time/session_count)/60;
fprintf("done p_number %d, avg time per session %0.2f minutes\n", p_number, time_minutes)
remaining_sessions = 194-session_count;
remaining_time = remaining_sessions * time_minutes;
hours_left = remaining_time/60;
fprintf("estimated time required for %d sessions: %0.2f hours\n\n", remaining_sessions, hours_left)


%% note: write to file commented out

% write the session 1 t2 values to file
for p_number = participant_numbers
    fname = sprintf("%s/p%d_s1_t2.txt", data_folder, p_number);
    % writematrx(participants_s1_t2{p_number}, fname)
end

% write the session 2 t2 values to file
for p_number = participant_numbers
    if ftonsets_2ses(p_number) == 1
        fname = sprintf("%s/p%d_s2_t2.txt", data_folder, p_number);
        % writematrx(participants_s2_t2{p_number}, fname)
    end
end



% write the session 1 maxt2 values to file
for p_number = participant_numbers
    fname = sprintf("%s/p%d_s1_maxt2.txt", data_folder,p_number);
    % writematrx(participants_s1_maxt2{p_number}, fname)
end

% write the session 2 maxt2 values to file
for p_number = participant_numbers
    if ftonsets_2ses(p_number) == 1
        fname = sprintf("%s/p%d_s2_maxt2.txt", data_folder,p_number);
        % writematrx(participants_s2_maxt2{p_number}, fname)
    end
end


% write the session 1 and 2 onset times to file
fname = sprintf("%s/participants_s1_onsets.txt", data_folder);
% writematrx(participants_s1_onsets, fname);


fname = sprintf("%s/participants_s2_onsets.txt", data_folder);
% writematrx(participants_s2_onsets, fname);

