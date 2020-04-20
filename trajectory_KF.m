clear all
clc
E = [];
cover = [];
T = [];
E1 = [];
for i = 1:2:25
i
% if i == 1
% table1 = csvread('D:\academics\Flight dataset\Flight1.csv',1);
% else 
% table1 = csvread('D:\academics\Flight dataset\Flight101.csv',1); 
% end
% data1 = table1(1:32:5177,2:4);

table1 = xlsread('D:\academics\Flight dataset\trajectory_data1.xlsx',i);
% data1 = table2array(table1);
data1 = table1;
table2 = xlsread('D:\academics\Flight dataset\trajectory_data1.xlsx',i+1);
% data2 = table2array(table2);
data2 = table2;
train_window = 15;
pred_start = train_window+3;
pred_horizon = 5;
y_pred_latitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
s2_pred_latitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
y_pred_longitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
s2_pred_longitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
y_pred_altitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
s2_pred_altitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
y_test_latitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
y_test_longitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
y_test_altitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
bounds_l_latitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
bounds_u_latitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
bounds_l_longitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
bounds_u_longitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
bounds_l_altitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
bounds_u_altitude = zeros(length(data1)-pred_start-pred_horizon+2,pred_horizon);
d = diff(data1);
d = [zeros(1,3);d];
data = [data1 d];
v = var(data);
vd = var(d);
B = zeros(6,6);
C = [1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0];
% Q = 1000*diag([v(1:3),vd]);
R = diag(v(1:3));
Q = diag([v(1:3),vd]);
% Q = 100*eye(6);
% R = 100*eye(3);
P = diag([v(1:3),vd]);
u = zeros(6,1);
x(:,1) = data(1,1:6)';
% x(:,1) = [0 0 0 ]';
for t = 1 : train_window+1
    A = [1 0 0 1 0 0; 0 1 0 0 1 0; 0 0 1 0 0 1; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1];
    x_prior = A*x(:,t);
    P_prior = A*P*A'+Q;
    K = P_prior*C'/(C*P_prior*C'+R);
    x(:,t+1) = x_prior+K*((data(t+1,1:3)'-C*x_prior));
    P = (eye(6)-K*C)*P_prior;
end
t = 1;
error = [];
cp = [];
t_list = [];
error_1step = [];
for pred_start = train_window+3 : length(data)-pred_horizon+1
    tic
    for m = 1:5
        A = [1 0 0 m 0 0; 0 1 0 0 m 0; 0 0 1 0 0 m; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1];
        x_pred(:,m) = A*x(:,pred_start-1)+B*u;
    end
    y_pred_latitude(t,:) = x_pred(1,:);
    y_pred_longitude(t,:) = x_pred(2,:);
    y_pred_altitude(t,:) = x_pred(3,:);
    x_prior  = x_pred(:,1);
    P_prior = A*P*A'+Q;
    b1 = y_pred_latitude(t,:)-1.96*sqrt(P_prior(1,1));
    b2 = y_pred_latitude(t,:)+1.96*sqrt(P_prior(1,1));
    b3 = y_pred_longitude(t,:)-1.96*sqrt(P_prior(2,2));
    b4 = y_pred_longitude(t,:)+1.96*sqrt(P_prior(2,2));
    b5 = y_pred_altitude(t,:)-1.96*sqrt(P_prior(3,3));
    b6 = y_pred_altitude(t,:)+1.96*sqrt(P_prior(3,3));
    K = P_prior*C'/(C*P_prior*C'+R);
    x(:,pred_start) = x_prior+K*((data(pred_start,1:3)'-C*x_prior));
    P = (eye(6)-K*C)*P_prior;
    e = abs(data(pred_start:pred_start+pred_horizon-1,1:3)-x_pred(1:3,:)');
    y_test_la = data(pred_start:pred_start+pred_horizon-1,1);
    y_test_lo = data(pred_start:pred_start+pred_horizon-1,2);
    y_test_al = data(pred_start:pred_start+pred_horizon-1,3);
    y_test_latitude(t,:) = y_test_la';
    y_test_longitude(t,:) = y_test_lo';
    y_test_altitude(t,:) = y_test_al';
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
    temp = sum(e.^2,2);
    error = [error;sqrt(mean(temp))];
    error_1step = [error_1step; sqrt(temp(1))];
    time = toc;
    cp = [cp;sum(c)/pred_horizon];
    t_list = [t_list;time];
    t = t+1;
end
E = [E;error];
E1 = [E1;error_1step];
cover = [cover;cp];
T = [T;t_list];
end
