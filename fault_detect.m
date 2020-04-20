function [y,G, GLRT] = fault_detect(e,sigma2,alpha)
%This is the function to implement the GLRT-based change-point detection method
GLRT = e^2/sigma2;
a = (e^2+sigma2)/sigma2;
b = (2*sigma2^2+e^2)/sigma2^2;
g = b/(2*a);
h = 2*a^2/b;
G = g*chi2inv(1-alpha,h);
if GLRT > G
    y = 1;
else 
    y = 0;
end