%
% fto_demo_loop_figures_VE2
%

set(groot,'defaultAxesToolbarVisible','off')


% changed corepath and function path (PJH)
% corepath = '/Volumes/fto2/';
corepath = '/home/phebden/Glasgow/DRP-3-code';
addpath([corepath,'/GAR/functions'])

% Note about plots: if figure NumberTitle is 'off', change to show titles

% this script will save plots here
plot_folder = 'ftonsets_plots_VE2';

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
Nperm = 1000; % 100 number of permutations -- use 1000 in non-demo scripts

s_num=1;   % <<<<<<<<<<<<<<<<<<<<<<<

s_number = s_num;

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


%% load virtual electrode's max t2 for session or session 2,  using VE2

if s_number == 1
    maxt2 = load('max_t2_avg_all_s1.csv');
    max_face_avg_session = load("max_face_avg_s1.csv");
    max_noise_avg_session = load("max_face_avg_s1.csv");
end 

if s_number == 2
    maxt2 = load('max_t2_avg_all_s2.csv');
    max_face_avg_session = load("max_face_avg_s2.csv");
    max_noise_avg_session = load("max_face_avg_s2.csv");
end 

%% COMPUTE PERMUTATION T-TESTS

% **************************************************************
% reshape each participant's avg face and noise erps
% to use for permutations and test for significance and cluster
% and the test cluster-sums for significance
% ***************************************************************

Ne=1;    % one virtual electrode per participant
Nf=451;  % number of time points
Ntf=75;  % 75 avg face ERPs
Ntn=75;  % 75 avg texture (noise) ERPs

maxt2_perm = zeros(Nf,Nperm);

% create random permutation samples
% use same samples for ttest and MI
indx_perm = zeros(Nperm, Ntf + Ntn);
for perm_iter = 1:Nperm % same trials used for all electrodes and time points
    indx_perm(perm_iter,:) = randperm(Ntf + Ntn);
end

% need to reshape so that it's 1 virtual electrode, 451 time pts, 75
% participants (instead of ~143 trials per electrode

ferp=reshape(max_face_avg_session,  [1,451,75]);
nerp=reshape(max_noise_avg_session, [1,451,75]);

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

% make the time points as -300 .. 600 milliseconds
Xf = 0:450;
Xf = Xf * 2;
Xf = Xf - 300;

figname = sprintf('Max t^2, permutations %d, s%d', Nperm, s_number);
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
legend("max t^2", "max t^2 permutations");   

fname=sprintf("%s/session_%d_permutations_VE2.png", plot_folder, s_number);
saveas(gcf,fname)

%% CLUSTER BASED STATISTICS

% baseline normalisation -- subtract 50th quantile of baseline T2
% changed hd (see functions folder) to Matlab's median, as suggested by GAR
for perm_iter = 1:Nperm
    temp=maxt2_perm(:,perm_iter) - median(maxt2_perm(Xf<0,perm_iter));
    temp(temp < 0)=0;
    maxt2_perm(:,perm_iter) = temp;
end
maxt2 = maxt2 - prctile(maxt2(Xf<0), 50);   % but prctile(x,50) == median(x) == hd(x)

maxt2(maxt2 < 0)=0;

if size(maxt2,1) == 1
    maxt2 = maxt2';
end



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

fprintf("Nperm=%d, using session s%d, onset %dms \n", Nperm, s_number, onset(1))



%% ILLUSTRATE ONSET RESULTS

onset = round(onset); % want integer
figname = sprintf('T CB onset {%d}ms: permutations %d, s%d', onset, Nperm, s_number);
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

fname=sprintf("%s/session_%d_permutations_clusters_VE2.png", plot_folder, s_number);
saveas(gcf,fname)


if close_all_figs == 1
    close all;  
end
    
% done session
 


   


