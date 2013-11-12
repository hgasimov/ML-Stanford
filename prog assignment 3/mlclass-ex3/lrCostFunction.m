function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%

h = sigmoid(X * theta); % h = sigmoid(transpose(theta) * X)

sum1 = y' * log(h);
sum2 = (1 - y)' * log(1 - h);
theta = theta(2:end); % Note that the first term grad(1)
                                % should not be regularized. 
reg_cost = lambda * sum(theta .^ 2) / (2*m);

J = - (sum1 + sum2) / m + reg_cost;

reg_grad = [0; lambda * theta / m]; % regularization term of the gradient. 
grad = X' * (h - y) / m + reg_grad;

% =============================================================


end
