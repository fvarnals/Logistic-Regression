function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

onesmatrix = ones(size(z));
powermatrix = ones(size(z)).*(1+e.^-z);
g = g + (onesmatrix./powermatrix);

end
