function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);

%calculate unregulated cost
J_unreg = (1/m) * (-y'*log(h) - (1-y)'*log(1-h));

%don't include theta(1) in regularization function
theta(1) = 0;

J = J_unreg  + ((lambda/(2*m))*(theta'*theta));

%calcuate gradient w/o regularization
grad = (1/m) * ((h-y)'*X); %gives 1X3 matrix

%add lambda regularization term
grad = grad + ((lambda/m).*theta');



% =============================================================

end
