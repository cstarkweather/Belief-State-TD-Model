function results = TD(x,O,T)
    
    % TD learning for partially observable MDP.
    % Author: Dr. Samuel J. Gershman
    %
    % USAGE: results = TD(x,O,T)
    %
    % INPUTS:
    %   x - stimulus timeseries (1 = nothing, 2 = stimulus, 3 = reward)
    %   O - [S x S x M] observation distribution: O(i,j,m) = P(x'=m|s=i,s'=j)
    %   T - [S x S] transition distribution: T(i,j) = P(s'=j|s=i)
    %
    % OUTPUTS:
    %   results - structure with the following fields:
    %               .w - weight vector at each time point (prior to updating)
    %               .b - belief state at each time point (prior to updating)
    %               .rpe - TD error (after updating)
    %
    %%
    % initialization
    S = size(T,1);      % number of states
    b = ones(S,1)/S;    % belief state
    w = zeros(S,1);     % weights
    
    % learning rates
    alpha = 0.1;          % value learning rate
    gamma = 0.98;
    %%
    for t = 1:length(x)

        b0 = b; % old posterior, used later
        b = b'*(T.*squeeze(O(:,:,x(t))));
        b=b';
        b = b./sum(b);

        % TD update
        w0 = w;
        r = double(x(t)==3);        % reward
        rpe = r + w'*(gamma*b-b0);  % TD error
        w = w + alpha*rpe*b0;         % weight update
        
        % store results
        results.w(t,:) = w0;
        results.b(t,:) = b0;
        results.rpe(t,1) = rpe;
        results.value(t,1) = w'*(b0); %estimated value
%         randcolor = rand;
%         subplot(2,1,1)
%         plot(b,'Color',[1-randcolor randcolor 1-randcolor])
%         hold on
%         subplot(2,1,2)
%         plot(sum(T.*squeeze(O(:,:,x(t)))),'Color',[1-randcolor randcolor 1-randcolor])
%         hold on
    end
