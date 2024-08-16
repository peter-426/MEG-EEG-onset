% fto_demo

% changed corepath and function path (PJH)
% corepath = '/Volumes/fto2/';
corepath = '/home/phebden/Glasgow/DRP-3-code/';
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
Nperm = 1000; % 100 number of permutations -- use 1000 in non-demo scripts

% generate plots for some or no participants
plot_p_number = 1;  % 0 for none, 1 for participant 1, or e.g. first 5

participant_numbers = 1:1;  % 120;  % <<<<=============-------------------
session_numbers     = 1:1;


%%

p_number = 1;
s_number = 1;
e_number = 40;

if s_number == 2 && ftonsets_2ses(p_number) == 0
    print("No session 2\n")
end

% load the data struct
fname_data = sprintf('%s/ftonsets_erps_avg/ftonsets_p%d_s%d',corepath, p_number, s_number);
load(fname_data); 

ferp = data.face_trials;  % 3D matrix electrodes x time points x trials
nerp = data.noise_trials; % 3D matrix electrodes x time points x trials
[Ne,Nf,Ntf] = size(ferp);
Ntn = size(nerp,3);
Xf = data.times;

f=figure();
f.Position = [100 100 640 800];

for ii=1:15
    y_row = ii*15;
    y_vals = ferp(1,:,ii) + y_row;
    plot(Xf, y_vals)
    hold on;
end
title("EEG face images", FontSize=22)
ylabel("trial", FontSize=20)
xlabel("time (ms)", FontSize=20)
yticks([])
xlim([min(Xf), max(Xf)])

fname=sprintf("%s/p%d_s%d_trials_ferps_electrode_%d.png", plot_folder, p_number, s_number, e_number);
saveas(gcf,fname)
  
%%

f=figure();
f.Position = [100 100 640 800];

for ii=1:15
    y_row = ii*15;
    y_vals = nerp(1,:,ii) + y_row;
    plot(Xf, y_vals)
    hold on;
end
title("EEG texture images", FontSize=22)
ylabel("trial", FontSize=20)
xlabel("time (ms)", FontSize=20)
yticks([])
xlim([min(Xf), max(Xf)])

fname=sprintf("%s/p%d_s%d_trials_nerps_electrode_%d.png", plot_folder, p_number, s_number, e_number);
saveas(gcf,fname)

%% plot erp for 1 electrode

electrode_num = 1;

% RED = FACE ERPs
% GREY = NOISE ERPs
% note the large N170 between 100-200 ms.
if p_number <= plot_p_number

    figname = sprintf('Trial ERPs for electrode number 1: p%d s%d', p_number, s_number);
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
fname=sprintf("%sp%d_s%d_trials_erps_electrode_%d.png", plot_folder, p_number, s_number, electrode_num);
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
fname=sprintf("%s/p%d_s%d_avg_erps_temp.png", plot_folder, p_number, s_number);
saveas(gcf,fname)
end

