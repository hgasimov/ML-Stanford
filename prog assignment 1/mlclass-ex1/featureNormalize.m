function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu = mean(X, 1);
sigma = std(X, 1);
mu_mat = ones(size(X, 1), 1) * mu;
sigma_mat = ones(size(X, 1), 1) * sigma;

X_norm = (X - mu_mat) ./ sigma_mat;

end
