%% overall specifications

max_lag_loadings = 3;           % maximum of factor loadings
    
%% indicator specifications
                             % Variable  % Category   %Trend    %Loadings  % Measuremnt Error Spec. 
spec.indicators = cell2table({'CORE_INF' ,'Infl'      ,[]       ,-1         ,'Exact'
                              'GDP'       ,'Real'     ,'RWd'    ,0          ,'Exact'; 
                              'INV'       ,'Real'     ,'RWd'    ,[1 0 -1]   ,'Error' ;
                              'IMP'       ,'Real'     ,'RWd'    ,[1 0 -1]   ,'Error' ;
                              'EXP'       ,'Real'     ,'RWd'    ,[1 0 -1]   ,'Error' ;
                              'UNEMP'     ,'Real'     ,'RW'     ,[1 0 -1]   ,'Error' ;
                              'CONSC'     ,'Real'     ,'RW'     ,[1 0 -1]   ,'Error' ;
                              'CAPU'      ,'Real'     ,'RW'     ,[1 0 -1]   ,'Error'; 
                              'INFEXP'    ,'Exp'      ,'RW'     ,[]         ,'Cycle'});    
                          
spec.indicators.Properties.VariableNames = {'Names','Category','Trend_Spec','Loadings','Measurement'};

% Trend:                    []: no trend; RW: Random walk; RWd: Random walk with drift
% Loadings:                 not yet implemented....
% Measuremnt Error Spec.:   Exact = no measuremant errors; Error = serially uncorrelated errors, Cycle = AR(2) errors

%% state specifications

spec.OutputGap = struct('IdentVariable','GDP','AR_lags',2,'Max_Loadings',max_lag_loadings,'CostPush_loadings',[]);   % leadings (+); laggings (-) 
spec.PC        = struct('AR_lags',1,'OutputGap_lags',3,'CostPush_lags',1);
spec.InflTrend = struct('AR_lags',1,'constant',1,'Trend','RWd');
spec.ImpInfl   = struct('AR_lags',2);



