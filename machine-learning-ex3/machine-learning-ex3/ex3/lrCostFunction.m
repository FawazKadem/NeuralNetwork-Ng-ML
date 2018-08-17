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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


%vectorized implementation of calculating theta' * x for all x.
z = X*theta;
g = sigmoid(z);
%g now = n-vector of sigmoided-predictions for all x parameterized by theta

% vectorized logistic regression cost function:
a = ((log(g))')*(-y);
b = ((-y) + 1); 
c = log((-g) + 1);
d = c' * b;

sumCost = sum(a - d);
J = sumCost ./ m;

regTemp = ((sum(theta(2:end).^2)) .* lambda) ./ (2.*m);

J = J + regTemp;
% vectorized logistic regression gradient function:
sumGrad = (X') * (g-y);
grad = sumGrad ./ m;

regGradTemp = theta .* lambda ./ m;
regGradTemp(1) = 0;

grad = grad + regGradTemp;









% =============================================================

grad = grad(:);

end
