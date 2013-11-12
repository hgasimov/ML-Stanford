function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.

X = [ones(m, 1) X];

% X ~ [m, n+1] matrix
% Theta1 ~ [25, n+1] matrix
% Theta2 ~ [10, 26] matrix

a2 = sigmoid(X * (Theta1')); % [m, 25] matrix
a2 = [ones(m, 1) a2]; % add a column of ones to the beginning of the matrix,
                      % so now a is a [m, 26] matrix
a3 = sigmoid(a2 * (Theta2')); % [m, 10] matrix

[maxval, p] = max(a3, [], 2); % take maximum of each row to predict most probable label



% =========================================================================


end
