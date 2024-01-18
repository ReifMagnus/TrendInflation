%% This script estimates a dynamic factor model with trends for non-stationary variables and a Philips-curve relation
clc; clear
%% housekeeping

estimate = 1;

addpath('functions');

spec.draws     = 25000;           % posterior draws
spec.burnin    = 10000;           % burn in
spec.thin      = 40;              % take every thin-th draw
spec.tau       = 30;              % length of training sample
spec.stability = 1;               % 1: only retain stable factor draws
spec.hmax      = 10;               % forecast horizon
spec.sv        = 'RW';            % t, RW, or normal
spec.tvp       = 0;

data_in = readtable('Data_inflation.xlsx','Sheet','data_DE');

load_model_extended_DE

% save data if desired
% estimation_date = datestr(now,'yyyy_mm_dd');
% if exist(['data/data_',estimation_date],'file') ~= 2
%     save(['data/data_',estimation_date],'data_in');
% end


data_raw        = data_in{1:end,2:end};
spec.categories = spec.indicators.Category;
spec.names      = data_in.Properties.VariableNames(2:end);
spec.trans_vec  = data_raw(1:2,ismember(spec.names,spec.indicators.Names));
spec.data_Q     = data_raw(3:end,ismember(spec.names,spec.indicators.Names));

%% organize dates
dates_short = data_in.Dates(2 + spec.tau + 1:end);

datesQ          = datetime(dates_short,'Format','QQQ-yyyy'); 
current_Q       = dateshift(datetime(datestr(now),'Format','QQQ-yyyy'),'start','quarter','current');               % current quarter

T_infl          = sum(isfinite(data_in{:,1})) - spec.tau;
date_last       = datetime(datesQ(T_infl),'Format','QQQ-yyyy');                % last observation quarter

if date_last < current_Q  
    spec.hmax   = spec.hmax +1 ;               % increase horizon by one if current quarter information is not available
end

dates_forecasts  = date_last + calquarters(1:spec.hmax)';
dates_total      = [datesQ(2:end-1); datesQ(end) +  calquarters(0:spec.hmax)'];                      % cut first observation due to taking differences
dates_total      = datetime(dates_total,'Format','QQQ-yyyy');

if date_last < current_Q  
    backcast_Q = find(datenum(dateshift(current_Q,'start','quarter','previous')) == datenum(dates_total));
else
    backcast_Q =[];
end

nowcast_Q   = find(datenum(current_Q) == datenum(dates_total));
forecast_Q  = find(datenum(dates_total) <= datenum(current_Q + calyears(2)) & datenum(dates_total) > datenum(current_Q));       % up to 2 years ahead

%% arrange data
spec.infl_ix  = find(strcmp(spec.categories,'Infl'));
spec.real_ix  = find(strcmp(spec.categories,'Real'));
spec.exp_ix   = find(strcmp(spec.categories,'Exp'));
spec.costp_ix = find(strcmp(spec.categories,'CostPush'));

spec.n_infl  = numel(spec.infl_ix);                                  % number of inflation indicators
spec.n_real  = numel(spec.real_ix);                                  % number of real indicators
spec.n_exp   = numel(spec.exp_ix);                                   % number of expectation indicators
spec.n_costp = numel(spec.costp_ix);                                 % number of cost-push indicators

spec.yraw                 = transform_ifo(spec.data_Q,spec.trans_vec');             % transform variables (according to excel-file)
spec.yraw(:,spec.infl_ix) = spec.yraw(:,spec.infl_ix)*4;                       % annualized inflation
spec.y0                   = spec.yraw(1:spec.tau,:);                                 % training sample
spec.y                    = spec.yraw(spec.tau+1:end,:);                               % estimation sample

[spec.T,spec.n] = size(spec.y);
spec.nan_id     = find(isnan(sum(spec.y,2)));

%% estimate model
if estimate == 1
    [results,posterior] = est_dfm(spec);
else
    load('Posteriors/Posterior_pre_covid_DE','posterior');  
    results = run_dfm(spec,posterior);
end
%% arrange output

% 1. forecasts
Yhat = results.y_level;                                                   % forecast of series in terms of input scale (levels)

% quaterly HICP level
Yhat_infl_level = Yhat(:,spec.infl_ix,:);

% q-o-q inflation in annualized terms
Yhat_infl_qoq      = (Yhat_infl_level(2:end,:,:)./Yhat_infl_level(1:end-1,:,:)-1)*400;
forecast_inflation = Yhat_infl_qoq(T_infl:end,:,:);

forecast_core_inflation     = squeeze(forecast_inflation(:,1,:));
if numel(spec.infl_ix) > 1
    forecast_headline_inflation = squeeze(forecast_inflation(:,2,:));
end

% 2. unobserved series
trend_inflation  = prctile(results.trend_inflation,[5,50,95],2);
output_gap       = prctile(results.output_gap,[5,50,95],2);
potential_output = squeeze(prctile(results.real_trends(:,1,:),[5,50,95],3));

%% figures
core_infl  = prctile(forecast_core_inflation,[5,20,50,80,95],2);
head_infl  = prctile(forecast_headline_inflation,[5,20,50,80,95],2);

n_fct      = size(core_infl,1);
light_grey = [224,224,224]./255;
dark_grey  = [96,96,96]./255;

light_blue = [102,102,255]./255;
dark_blue  = [051 051 255]./255;

plot_core       = [infl(1:T_infl - 1,1,3); nan(size(forecast_inflation,1),1)];               % ex-post data
plot_core_fcast = [repmat(infl(1:T_infl - 1,1,3),1,5); core_infl];     % median forecast

plot_head       = [infl(1:T_infl - 1 ,2,3); nan(size(forecast_inflation,1),1)];               % ex-post data
plot_head_fcast = [repmat(infl(1:T_infl - 1,2,3),1,5); head_infl];     % median forecast


figure('units','normalized','pos',[.1 .1 .5 .35])
jbfill(dates_total(end-16:end)',plot_core_fcast(end-16:end,1)',plot_core_fcast(end-16:end,5)',dark_blue,dark_blue,0,0.3);  hold on  
jbfill(dates_total(end-16:end)',plot_core_fcast(end-16:end,2)',plot_core_fcast(end-16:end,4)',light_blue,light_blue,0,0.3);  hold on  
p2 = plot(dates_total(end-16:end),plot_core(end-16:end),'k-x','Linewidth',2);
p1 = plot(dates_total(end-16:end),[nan(16-n_fct,1); plot_core_fcast(end-n_fct:end,3)],'r--x','Linewidth',2);  hold on
yline(2,'k--','Linewidth',.25);
grid on; title('Forecast - core inflation');
box off

figure('units','normalized','pos',[.1 .1 .5 .35])
jbfill(dates_total(end-16:end)',plot_head_fcast(end-16:end,1)',plot_head_fcast(end-16:end,5)',dark_blue,dark_blue,0,0.3);  hold on  
jbfill(dates_total(end-16:end)',plot_head_fcast(end-16:end,2)',plot_head_fcast(end-16:end,4)',light_blue,light_blue,0,0.3);  hold on  
p2 = plot(dates_total(end-16:end),plot_head(end-16:end),'k-x','Linewidth',2);
p1 = plot(dates_total(end-16:end),[nan(16-n_fct,1); plot_head_fcast(end-n_fct:end,3)],'r--x','Linewidth',2);  hold on
yline(2,'k--','Linewidth',.25);hold on
grid on; title('Forecast - headline inflation');
box off


figure('units','normalized','pos',[.1 .1 .5 .35])
jbfill(dates_total(1:length(trend_inflation))',trend_inflation(:,1)',trend_inflation(:,3)',light_blue,light_blue,0,0.3);  hold on  
p1 = plot(dates_total(1:length(trend_inflation)),trend_inflation(:,2),'r-','Linewidth',2);  hold on
p2 = plot(dates_total(1:length(trend_inflation)),fillmissing(data_raw(spec.tau+3:end,end),'previous'),'k-','Linewidth',2);  hold on
% p4 = plot(dates_total(1:length(trend_inflation)),results.states_mean(:,end),'b-','Linewidth',2);  hold on
yline(2,'k--','Linewidth',.25);
grid on; title('Trend inflation'); legend([p1 p2],'Trend-Inflation','Inflation Expectations');
box off


figure('units','normalized','pos',[.1 .1 .5 .35])
jbfill(dates_total(1:length(output_gap))',output_gap(:,1)',output_gap(:,3)',light_blue,light_blue,0,0.3);  hold on  
p1 = plot(dates_total(1:length(output_gap)),output_gap(:,2),'r-','Linewidth',2);  hold on
yline(0,'k--','Linewidth',.25);hold on
grid on; title('Output gap');
box off

figure('units','normalized','pos',[.1 .1 .5 .35])
jbfill(dates_total(1:length(potential_output))',potential_output(:,1)',potential_output(:,3)',light_blue,light_blue,0,0.3);  hold on  
p1 = plot(dates_total(1:length(potential_output)),potential_output(:,2),'r-','Linewidth',2);  hold on
grid on; title('Potential output');
box off
 
figure('units','normalized','pos',[.1 .1 .5 .35])
p1 = plot(dates_total,results.states(1:length(dates_total),12,3),'r-','Linewidth',2);  hold on
p1 = plot(dates_total,results.states(1:length(dates_total),13,3),'k-','Linewidth',2);  hold on
p2 = plot(dates_total(1:T_infl),data_in{spec.tau+3:end,end-1},'b-','Linewidth',1.5);  hold on
yline(0,'k--','Linewidth',.25); hold on
grid on; title('Oil Decomposition'); legend('Trend','Cycle','Actual')
box off

