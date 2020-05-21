clear all
clc
% Multi-step error
E = [];
% One-step error
E1 = [];
% Multi-step coverage probability
C = [];
% Computational time
T = [];
for i = 1:2:25
(i+1)/2
% read and process data
table1 = xlsread('D:\academics\Flight dataset\trajectory_data1.xlsx',i);
data1 = table1;
table2 = xlsread('D:\academics\Flight dataset\trajectory_data1.xlsx',i+1);
data2 = table2;
v = diff(data1);
v = [zeros(1,3);v];
data = [data1 data2];
train_window = 15;
pred_horizon = 5;
data_max = max(data);
data_min = min(data);
data_scaled = (data-repmat(data_min,length(data),1))./(repmat(data_max,length(data),1)-repmat(data_min,length(data),1));
data1_pred = data1 + v;
data1_pred(end,:) = [];
data1_pred = [zeros(1,3);data1_pred];
pred_error = data1 - data1_pred;
% Hyper-parameter initialization
mf = {@meanZero};
cf = {@covSEard};
lf = {@likGauss};
inf = {@infGaussLik};
sf = 1;
sn = 0.36;
hyp0.mean = []; 
hyp0.lik = log(sn);
Ncg = 500;

%Model parameter initialization
t = 1;
alpha = 0.05;
epsilon = 0.2;
pred_start = train_window+3;
y_pred_latitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
s2_pred_latitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
y_pred_longitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
s2_pred_longitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
y_pred_altitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
s2_pred_altitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
y_test_latitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
y_test_longitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
y_test_altitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
bounds_l_latitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
bounds_u_latitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
bounds_l_longitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
bounds_u_longitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
bounds_l_altitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);
bounds_u_altitude = zeros(length(data_scaled)-pred_start-pred_horizon+2,pred_horizon);

error = [];
error_1step = [];
cp = [];
t_list = [];
e1_pred = 10000;
e2_pred = 10000;
e3_pred = 10000;
sigma2_1 = 1;
sigma2_2 = 1;
sigma2_3 = 1;
X_train_la = data_scaled(2:pred_start-2,1:6);
y_train_la = pred_error(3:pred_start-1,1);
X_train_lo = data_scaled(2:pred_start-2,1:6);
y_train_lo = pred_error(3:pred_start-1,2);
X_train_al = data_scaled(2:pred_start-2,1:6);
y_train_al = pred_error(3:pred_start-1,3);

ell = ones(6,1);
hyp0.cov = log([ell;sf]); 
hyp_la = minimize(hyp0 , @gp, -Ncg , inf , mf , cf , lf , X_train_la, y_train_la);
hyp_lo = minimize(hyp0 , @gp, -Ncg , inf , mf , cf , lf , X_train_lo, y_train_lo);
hyp_al = minimize(hyp0 , @gp, -Ncg , inf , mf , cf , lf , X_train_al, y_train_al);

for pred_start = train_window+3 : length(data_scaled)-pred_horizon+1
    tic
    % change-point detection
    e1_det = fault_detect(e1_pred,sigma2_1,alpha);
    if e1_det
        ell = ones(6,1);
        hyp0.cov = log([ell;sf]); 
        hyp_la = minimize(hyp0 , @gp, -Ncg , inf , mf , cf , lf , X_train_la, y_train_la);
    end
    e2_det = fault_detect(e2_pred,sigma2_2,alpha);
    if e2_det
        ell = ones(6,1);
        hyp0.cov = log([ell;sf]); 
        hyp_lo = minimize(hyp0 , @gp, -Ncg , inf , mf , cf , lf , X_train_lo, y_train_lo);
    end
    e3_det = fault_detect(e3_pred,sigma2_3,alpha);
    if e3_det
        ell = ones(6,1);
        hyp0.cov = log([ell;sf]); 
        hyp_al = minimize(hyp0 , @gp, -Ncg , inf , mf , cf , lf , X_train_al, y_train_al);
    end
    
    
    v_la = v(pred_start-1,1);
    v_lo = v(pred_start-1,2);
    v_al = v(pred_start-1,3);
    X_test_la = data_scaled(pred_start-1,1:6);
    X_test_lo = data_scaled(pred_start-1,1:6);
    X_test_al = data_scaled(pred_start-1,1:6);
    
    % Collect training data and train GPR models
    [X_train_temp, y_train_temp] = DHMP(hyp_la, inf, mf, cf, lf, X_train_la, y_train_la, X_test_la, epsilon);
    [y_pred_la, s2_pred_la] = gp(hyp_la, inf, mf , cf , lf,  X_train_temp, y_train_temp, X_test_la);
    X_train_la = [X_train_temp;X_test_la];
    y_train_la = [y_train_temp;pred_error(pred_start,1)];
    [beta_la, gamma_la] = get_beta_gamma(hyp_la, mf, cf, lf, X_train_temp, y_train_temp, repmat(X_test_la,5,1), 6, 1e-1, 0.04, 0.01);
    [X_train_temp, y_train_temp] = DHMP(hyp_lo, inf, mf, cf, lf, X_train_lo, y_train_lo, X_test_lo, epsilon);
    [y_pred_lo, s2_pred_lo] = gp(hyp_lo, inf, mf , cf , lf,  X_train_temp, y_train_temp, X_test_lo);
    X_train_lo = [X_train_temp;X_test_lo];
    y_train_lo = [y_train_temp;pred_error(pred_start,2)];
    [beta_lo, gamma_lo] = get_beta_gamma(hyp_lo, mf, cf, lf, X_train_temp, y_train_temp, repmat(X_test_lo,5,1), 6, 1e-1, 0.04, 0.01);
    [X_train_temp, y_train_temp] = DHMP(hyp_al, inf, mf, cf, lf, X_train_al, y_train_al, X_test_al, epsilon);
    [y_pred_al, s2_pred_al] = gp(hyp_al, inf, mf , cf , lf,  X_train_temp, y_train_temp, X_test_al);
    X_train_al= [X_train_temp;X_test_al];
    y_train_al = [y_train_temp;pred_error(pred_start,3)];
    [beta_al, gamma_al] = get_beta_gamma(hyp_al, mf, cf, lf, X_train_temp, y_train_temp, repmat(X_test_al,5,1), 6, 1e-1, 0.04, 0.01);
    
    i = 1:pred_horizon;
    v1_test = i*v(pred_start-1,1);
    v2_test = i*v(pred_start-1,2);
    v3_test = i*v(pred_start-1,3);
    
    % Make predictions
    y_pred_latitude(t,:) = data1(pred_start-1,1)+v1_test+y_pred_la;
    s2_pred_latitude(t,:) = ones(1,pred_horizon)*s2_pred_la;
    y_test_la = data1(pred_start:pred_start+pred_horizon-1,1);
    y_test_latitude(t,:) = y_test_la';
    y_pred_longitude(t,:) = data1(pred_start-1,2)+v2_test+y_pred_lo;
    s2_pred_longitude(t,:) = ones(1,pred_horizon)*s2_pred_lo;
    y_test_lo = data1(pred_start:pred_start+pred_horizon-1,2);
    y_test_longitude(t,:) = y_test_lo';
    y_pred_altitude(t,:) = data1(pred_start-1,3)+v3_test+y_pred_al;
    s2_pred_altitude(t,:) = ones(1,pred_horizon)*s2_pred_al;
    y_test_al = data1(pred_start:pred_start+pred_horizon-1,3);
    y_test_altitude(t,:) = y_test_al';
    
    % Calculate predictive errors
    e1 = abs(y_pred_latitude(t,:)'-y_test_la);
    e1_pred = e1(1);
    sigma2_1 = exp(hyp_la.lik);
    
    e2 = abs(y_pred_longitude(t,:)'-y_test_lo);
    e2_pred = e2(1);
    sigma2_2 = exp(hyp_lo.lik);
   
    e3 = abs(y_pred_altitude(t,:)'-y_test_al);
    e3_pred = e3(1);
    sigma2_3 = exp(hyp_al.lik);
   
    temp = mean(e1.^2+e2.^2+e3.^2);
    error = [error; sqrt(temp)];
    temp = sqrt(e1(1)^2+e2(1)^2+e3(1)^2);
    error_1step = [error_1step; temp];
    time = toc;
%     b1 = y_pred_latitude(t,:)-1.96*sqrt(s2_pred_latitude(t,:));
%     b2 = y_pred_latitude(t,:)+1.96*sqrt(s2_pred_latitude(t,:));
%     b3 = y_pred_longitude(t,:)-1.96*sqrt(s2_pred_longitude(t,:));
%     b4 = y_pred_longitude(t,:)+1.96*sqrt(s2_pred_longitude(t,:));
%     b5 = y_pred_altitude(t,:)-1.96*sqrt(s2_pred_altitude(t,:));
%     b6 = y_pred_altitude(t,:)+1.96*sqrt(s2_pred_altitude(t,:));

    % Build confidence inervals
    b1 = y_pred_latitude(t,:)-sqrt(beta_la)*sqrt(s2_pred_latitude(t,:))-gamma_la;
    b2 = y_pred_latitude(t,:)+sqrt(beta_la)*sqrt(s2_pred_latitude(t,:))+gamma_la;
    b3 = y_pred_longitude(t,:)-sqrt(beta_lo)*sqrt(s2_pred_longitude(t,:))-gamma_lo;
    b4 = y_pred_longitude(t,:)+sqrt(beta_lo)*sqrt(s2_pred_longitude(t,:))+gamma_lo;
    b5 = y_pred_altitude(t,:)-sqrt(beta_al)*sqrt(s2_pred_altitude(t,:))-gamma_al;
    b6 = y_pred_altitude(t,:)+sqrt(beta_al)*sqrt(s2_pred_altitude(t,:))+gamma_al;
    for j = 1:pred_horizon
        if y_test_la(j)>=b1(j) && y_test_la(j)<=b2(j) && y_test_lo(j)>=b3(j) && y_test_lo(j)<=b4(j) && y_test_al(j)>=b5(j) && y_test_al(j)<=b6(j)
            c(j) = 1;
        else 
            c(j) = 0;
        end
    end
    bounds_l_latitude(t,:) = b1;
    bounds_u_latitude(t,:) = b2;
    bounds_l_longitude(t,:) = b3;
    bounds_u_longitude(t,:) = b4;
    bounds_l_altitude(t,:) = b5;
    bounds_u_altitude(t,:) = b6;
    cp = [cp;sum(c)/pred_horizon];
    t_list = [t_list;time];
    t = t+1;
end
E = [E;error];
E1 = [E1;error_1step];
C = [C;cp];
T = [T;t_list];
end

