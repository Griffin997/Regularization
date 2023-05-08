%%This is a file with the 

%Example clculation for J
t = [1, 2, 3, 4];
con1 = 0.5;
con2 = 0.7;
tau1 = 2;
tau2 = 3;
jacobian = J(t, con1, con2, tau1, tau2) % returns a 4x4 matrix

%Example calculation for GCV
params = [0.4,0.6,50,70];
RSS = 0.1;
lamb = 0.1;
tdata = linspace(8,32*8,32);
GCV = get_GCV_value(tdata, params, RSS, lamb) % returns some numeric value

function jacobian = J(t, con1, con2, tau1, tau2)
% Computes the Jacobian matrix for a given set of time data and parameters. The Jacobian matrix contains the partial
% derivatives of the function with respect to each of the parameters.
% 
% Usage: jacobian = J(t, con1, con2, tau1, tau2)
%
% Arguments:
% - t: A vector containing the time data
% - con1: A scalar representing a parameter value
% - con2: A scalar representing another parameter value
% - tau1: A scalar representing yet another parameter value
% - tau2: A scalar representing one more parameter value
%
% Returns:
% - jacobian: A matrix containing the partial derivatives of the function with respect to each of the parameters. 
%             The size of the matrix is length(t) x 4.
    func1 = exp(-t/tau1);
    func2 = exp(-t/tau2);
    func3 = (con1*t).*exp(-t/tau1)/(tau1^2);
    func4 = (con2*t).*exp(-t/tau2)/(tau2^2);
    jacobian = [func1; func2; func3; func4]';
end

function GCV = get_GCV_value(tdata, params, RSS, lamb)
% Computes the generalized cross-validation (GCV) value for a given set of parameters, residual sum of squares (RSS),
% regularization parameter lambda, and time data.% 
% Usage: GCV = get_GCV_value(params, RSS, lamb, tdata)
%
% Arguments:
% - tdata: A vector containing the time data
% - params: A cell array of parameter values
% - RSS: The residual sum of squares between the estimated curve and the data
%       Example calculation: RSS = sum((est_curve - data).^2);
% - lamb: The regularization parameter
%
% Returns:
% - GCV: The generalized cross-validation value for the given set of parameters

    wmat = [1,0,0,0; 0,1,0,0; 0,0,0.01,0; 0,0,0,0.01];
    GCVjacobian = J(tdata, params(1), params(2), params(3), params(4));
    C_GCV = GCVjacobian*inv(GCVjacobian'*GCVjacobian+(lamb^2)*wmat'*wmat)*GCVjacobian';
    [n,~] = size(C_GCV);
    identity = eye(n);
    GCVdenominator = (trace(identity - C_GCV))^2;
    GCV = RSS/GCVdenominator;
end
