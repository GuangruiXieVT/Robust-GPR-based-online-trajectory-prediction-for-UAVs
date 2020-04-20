function [beta,gamma] = get_beta_gamma(hyp, mf, cf, lf, x, y_bar, xs, E, tau, delta, deltaL)
% This is the function to calculate beta and gamma in the uniform error
% bounds for GPR
[post, ~, ~] = infGaussLik(hyp, mf, cf, lf, x, y_bar);
ls = exp(hyp.cov(1:E));
sf = exp(hyp.cov(E+1));
Lk = norm(sf^2*exp(-0.5)./ls);
[Ntr,~] = size(x);
[Nte,~] = size(xs);
Lnu = Lk*sqrt(Ntr)*norm(post.alpha);
omega = sqrt(2*tau*Lk*(1+Ntr*norm(post.Kinv)*sf^2));
beta = 2*log((1+((max(max(xs))-min(min(xs)))/tau))^E/delta);
x = x';
xs = xs';
k = @(x,xp) sf^2 * exp(-0.5*sum((x-xp).^2./ls.^2,1));
dkdxi = @(x,xp,i)  -(x(i,:)-xp(i,:))./ls(i)^2 .* k(x,xp);
ddkdxidxpi = @(x,xp,i) ls(i)^(-2) * k(x,xp) +  (x(i,:)-xp(i,:))/ls(i)^2 .*dkdxi(x,xp,i);
dddkdxidxpi = @(x,xp,i) -ls(i)^(-2) * dkdxi(x,xp,i) - ls(i)^(-2) .*dkdxi(x,xp,i) ...
    +  (x(i,:)-xp(i,:))/ls(i)^2 .*ddkdxidxpi(x,xp,i);

r = max(pdist(xs')); Lfs = zeros(E,1);
for e=1:E
    maxk = max(ddkdxidxpi(xs,xs,e));
    Lkds = zeros(Nte,1);
    for nte = 1:Nte
       Lkds(nte) = max(dddkdxidxpi(xs,xs(:,nte),e));
    end
    Lkd = max(Lkds);  
    Lfs(e) = sqrt(2*log(2*E/deltaL))*maxk + 12*sqrt(6*E)*max(maxk,sqrt(r*Lkd));
end
Lfh =  norm(Lfs);
gamma = tau*(Lnu+Lfh) + sqrt(beta)*omega;
