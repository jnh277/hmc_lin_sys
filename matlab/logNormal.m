function [lpdf, glpdf] = logNormal(x, mu, sigma)
z = (x - mu)./sigma;
n = length(x);
% lpdf = sum(-log(sigma) - .5*log(2*pi) - .5*(z.^2));

% lpdf = -sum(.5*(z.^2))-n*log(sigma) - .5*n*log(2*pi);
lpdf = -n*log(sigma)-0.5*n*log(2*pi) - 0.5*((x-mu).'*(x-mu))/sigma^2;

glpdf = -z./sigma;

end

