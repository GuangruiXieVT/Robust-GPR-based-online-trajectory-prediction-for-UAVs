function [X_train, y_train] = DHMP(hyp, inf, mf, cf, lf, X, y, xs, epsilon)
% This is the function to implement DHMP for GPR
try
[ymu, ys2] = gp(hyp, inf, mf , cf , lf,  X, y, xs);
while length(X)>= 3
for i = 1:length(X)
    d = zeros(length(X),1);
    X_temp = X;
    y_temp = y;
    X_temp(i,:) = [];
    y_temp(i) = [];
    [ymu1, ys21] = gp(hyp, inf, mf , cf , lf,  X_temp, y_temp, xs);
    sigbar = (diag(ys2)+diag(ys21))/2;
    d2(i) = 1-det(diag(ys2))^(1/4)*det(diag(ys21))^(1/4)*exp(-1/8*(ymu-ymu1)'*inv(sigbar)*(ymu-ymu1))/det(sigbar);
end
d_min = min(d);
idx = find(d==d_min);
if d_min <= epsilon
    X(idx(1),:) = [];
    y(idx(1)) = [];
else
    break
end
end
X_train = X;
y_train = y;
catch 
    X_train = X;
    y_train = y;
end
