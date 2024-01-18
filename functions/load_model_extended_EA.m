%% overall specifications

max_lag_loadings = 3;           % maximum of factor loadings
    
%% indicator specifications
                              % Name      % Category  % Trend Spec  % Loadings  % Error Spec  % prior shares of variance of y explained by trend
spec.indicators = cell2table({'CORE_INF'  ,'Infl'     ,[]           ,[1 0]          ,'Exact'      ,.5;
                              'HEAD_INF'  ,'Infl'     ,[]           ,[1 0]          ,'Exact'      ,.5;    
                              'GDP'       ,'Real'     ,'RWd'        ,0         ,'Exact'      ,.5;     
                              'INV'       ,'Real'     ,'RWd'        ,[1 0 -1]   ,'Error'      ,.5;    
                              'IMP'       ,'Real'     ,'RWd'        ,[1 0 -1]   ,'Error'      ,.5;    
                              'EXP'       ,'Real'     ,'RWd'        ,[1 0 -1]   ,'Error'      ,.5;    
                              'UNEMP'     ,'Real'     ,'RW'         ,[1 0 -1]   ,'Error'      ,.5;    
                              'CONSC'     ,'Real'     ,'RW'         ,[1 0 -1]   ,'Error'      ,.5;    
                              'CAPU'      ,'Real'     ,'RW'         ,[1 0 -1]   ,'Error'      ,.5;     
                              'OIL'       ,'CostPush' ,'RW'         ,[]         ,'Exact'      ,.5;                              
                              'INFEXP'    ,'Exp'      ,'RW'         ,[]         ,'Error'      ,.5});    
                          
spec.indicators.Properties.VariableNames = {'Names','Category','Trend_Spec','Loadings','Measurement','Trend_share'};

% check whether each variable is loaded
if length(data_in.Properties.VariableNames(2:end)) ~= size(spec.indicators,1)   
    i = 1;   
    while i <= size(spec.indicators,1)   
        try
            isVariable(i) = sum(strcmp(spec.indicators.Names{i},data_in.Properties.VariableNames));
            i = i + 1;
        catch
            i = i + 1;
        end
    end
    if sum(~isVariable) ~= 0,warning('Some variables are not loaded'); end
    spec.indicators(~isVariable,:) = [];    
end


% Trend:                    []: no trend; RW: Random walk; RWd: Random walk with drift
% Loadings:                 not yet implemented....
% Measuremnt Error Spec.:   Exact = no measuremant errors; Error = serially uncorrelated errors, Cycle = AR(2) errors                 

%% state specifications

spec.OutputGap = struct('IdentVariable','GDP','AR_lags',2,'Max_Loadings',max_lag_loadings,'CostPush_loadings',[]);   % leadings (+); laggings (-) 
spec.PC        = struct('AR_lags',1,'OutputGap_lags',3,'CostPush_lags',3);
spec.InflTrend = struct('AR_lags',1,'constant',1,'Trend','RWd');
spec.CostPush  = struct('AR_lags',2,'OutputGap_lags',0);

%% prior specifications

% factor model for all real indicators
spec.prior.loadings.tightness = .5;                                                      % thightness factor, see D'Agostino et al.
spec.prior.loadings.decay     = 1;
spec.prior.loadings.mean      = zeros(spec.OutputGap.Max_Loadings,1);

% inflation expectations
spec.prior.infexp.mean     = [0; 1];
spec.prior.infexp.variance = diag([.1^2  .1^2]);

% cost-push trend
spec.prior.costp_trend.mean     = [0; 1];                                            % shrink towards driftless random walk
spec.prior.costp_trend.variance = eye(2)*0.001;

% output-gap
spec.prior.output_gap.mean     = [1.352; -0.508];                                            % shrink towards driftless random walk
spec.prior.output_gap.variance = [0.0806 -0.0597; -0.0597 0.0464];   % numbers taken from JL;

% trend inflation
spec.prior.inftrend.mean     = [0; 1];                                     % shrink towards driftless random walk
spec.prior.inftrend.variance = diag([0.001^2 0.001^2]);

% Core Phillips-curve: means
spec.prior.pc.output_gap.mean    = zeros(spec.PC.OutputGap_lags,1);                                                          % priors on output gap in PC equations
spec.prior.pc.output_gap.mean(2) = .5;                                                                     % prior on first lag of output gap
spec.prior.pc.infl_gap.mean      = zeros(spec.PC.AR_lags,1);                                                          % prior on inflation gap
spec.prior.pc.costp_gap.mean     = zeros(spec.PC.CostPush_lags,1);

% Core Phillips-curve: variances
spec.prior.pc.output_gap.variance    = ones(spec.PC.OutputGap_lags,1)*.01^2;                                   % tightly centered around zero
spec.prior.pc.output_gap.variance(2) = 1;                                                                      % first lag of output gap
spec.prior.pc.infl_gap.variance      = ones(spec.PC.AR_lags,1)*.1^2;
spec.prior.pc.infl_gap.variance(1)   = .1^2;
spec.prior.pc.costp_gap.variance     = ones(spec.PC.CostPush_lags,1)*.01^2;                                % tightly centered around zero

% Headline Phillips-curve: means
spec.prior.pc_head.output_gap.mean    = zeros(spec.PC.OutputGap_lags,1);                                                          % priors on output gap in PC equations
spec.prior.pc_head.infl_gap.mean      = zeros(spec.PC.AR_lags,1);                                                          % prior on inflation gap
spec.prior.pc_head.costp_gap.mean     = zeros(spec.PC.CostPush_lags,1);                                                          % prior on inflation gap

% Headline Phillips-curve: variances
spec.prior.pc_head.output_gap.variance = ones(spec.PC.OutputGap_lags,1)*10^-9;                                % tightly centered around zero
spec.prior.pc_head.infl_gap.variance   = ones(spec.PC.AR_lags,1)*.5^2;
spec.prior.pc_head.costp_gap.variance  = ones(spec.PC.CostPush_lags,1)*.5^2;                                                          % prior on inflation gap

% Cost-push process 
spec.prior.cost_push.output_gap.mean   = zeros(spec.CostPush.OutputGap_lags,1);
spec.prior.cost_push.costpush_gap.mean = zeros(spec.CostPush.AR_lags,1);

spec.prior.cost_push.output_gap.variance   = .1*ones(spec.CostPush.OutputGap_lags,1);
spec.prior.cost_push.costpush_gap.variance = (.5./((1:spec.CostPush.AR_lags).^2))';

% cost-push error cycle
spec.prior.cost_push_error.mean     = [0; 0];
spec.prior.cost_push_error.variance = eye(2)*0.1^2;

% inflation expecations error cycle
spec.prior.infexp_error.mean     = [0; 0];
spec.prior.infexp_error.variance = eye(2)*0.1^2;

%------------------------------------------------------------------------------------------------------------------
% shock variances (standard priors for the shock variances, if left commented priors from Jarocinski/Lenza are used)
% spec.prior.shocks.df_prior = 10;                                                  % degrees of freedom

% scale coefficients
% measurement equations
% spec.prior.shocks.scale.real_indicators = 0.09;
% spec.prior.shocks.scale.inflation       = 0.09;
% spec.prior.shocks.scale.inf_exp         = 0.09;
% spec.prior.shocks.scale.costp           = 0.09;
% 
% spec.prior.shocks.mean.real_indicators = spec.prior.shocks.scale.real_indicators/(spec.prior.shocks.df_prior-1);
% spec.prior.shocks.mean.inflation       = spec.prior.shocks.scale.inflation/(spec.prior.shocks.df_prior-1);
% spec.prior.shocks.mean.inf_exp         = spec.prior.shocks.scale.inf_exp/(spec.prior.shocks.df_prior-1);
% spec.prior.shocks.mean.costp           = spec.prior.shocks.scale.costp/(spec.prior.shocks.df_prior-1);

 % transition equations
% spec.prior.shocks.scale.output_gap      = 0.09;
% spec.prior.shocks.scale.real_trends     = 0.09;
% spec.prior.shocks.scale.inf_gap         = 0.09;
% spec.prior.shocks.scale.inf_trend       = 0.09;
% spec.prior.shocks.scale.costp_gap       = 0.6;
% spec.prior.shocks.scale.costp_trend     = 0.001;
% spec.prior.shocks.scale.inf_exp_cyc     = 0.09;
% spec.prior.shocks.scale.costp_cyc       = 0.09;
% 
% spec.prior.shocks.mean.output_gap      = spec.prior.shocks.scale.output_gap/(spec.prior.shocks.df_prior-1);
% spec.prior.shocks.mean.real_trends     = spec.prior.shocks.scale.real_trends/(spec.prior.shocks.df_prior-1);
% spec.prior.shocks.mean.inf_gap         = spec.prior.shocks.scale.inf_gap/(spec.prior.shocks.df_prior-1);
% spec.prior.shocks.mean.inf_trend       = spec.prior.shocks.scale.inf_trend/(spec.prior.shocks.df_prior-1);
% spec.prior.shocks.mean.costp_gap       = spec.prior.shocks.scale.costp_gap/(spec.prior.shocks.df_prior-1);
% spec.prior.shocks.mean.costp_trend     = spec.prior.shocks.scale.costp_trend/(spec.prior.shocks.df_prior-1);
% spec.prior.shocks.mean.inf_exp_cyc     = spec.prior.shocks.scale.inf_exp_cyc/(spec.prior.shocks.df_prior-1);
% spec.prior.shocks.mean.costp_cyc       = spec.prior.shocks.scale.costp_cyc/(spec.prior.shocks.df_prior-1);


if sum(strcmp(spec.indicators.Category,'CostPush')) == 0
 [spec.prior.pc.costp_gap.mean,spec.prior.pc_head.costp_gap.mean,spec.prior.pc.costp_gap.variance,spec.prior.pc_head.costp_gap.variance] = deal([]);
end