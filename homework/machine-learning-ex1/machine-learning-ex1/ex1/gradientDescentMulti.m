function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    theta = theta - (alpha/m)*X'*((X*theta)-y);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
%     temp = zeros(1,size(X,2));
%     for i=1:size(X,2)
%         temp(1,i) = sum(((X*theta)-y).*X(:,i));
%     end
%     theta = theta - (alpha/m)*temp';
    %theta = theta - (alpha/m)*[sum(((X*theta)-y)),sum(((X*theta)-y).*X(:,2)),sum(((X*theta)-y).*X(:,3))]';
end

end
