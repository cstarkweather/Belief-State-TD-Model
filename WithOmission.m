% Dopamine RPEs Reflect Hidden State Inference Across Time
% Task 2 Model (90% Rewarded)
% POMDP in the spirit of Daw et al. (2006), Rao (2010)
% Authors: Clara Kwon Starkweather, Dr. Samuel J. Gershman


%initialize
    nTrials = 1000; %number of trials
    x=[]; %series of observations during trials - will be filled in later

% Gaussian probability distribution and cumulative probability distribution
    ISIpdf=normpdf([1.2 1.4 1.6 1.8 2 2.2 2.4 2.6 2.8],2,0.5)/sum(normpdf([1.2 1.4 1.6 1.8 2 2.2 2.4 2.6 2.8],2,0.5));
    ISIcdf=[0.0477    0.1312    0.2558    0.4142    0.5858    0.7442    0.8688    0.9523    1.0000];
 
% Uncomment 2 lines directly below for flat probability distribution -
% as seen in Supplementary Figure 9
%    ISIpdf=[0.111 0.111 0.111 0.111 0.111 0.111 0.111 0.111 0.111];
%    ISIcdf=[0.111 0.222 0.333 0.444 0.555 0.666 0.777 0.888 1.0];

% Create distribution of ISI's - 10% unassigned are omission trials
% Possible ISIs range from 5-13
for i=1:nTrials-nTrials/10
    ISIdistributionMatrix(i)=sum(ISIcdf<rand)+5;
end
ISIdistributionMatrix=[ISIdistributionMatrix NaN(1,nTrials/10)]; %omission trials are marked as 'NaN'
ISIdistributionMatrix=ISIdistributionMatrix(randperm(length(ISIdistributionMatrix))); %Randomize when the omission trials occur

% Calculate hazard rate of receiving reward after substates 5-14 (ISIhazard);
% Used later to create transition matrix
    ISIhazard(1)=ISIpdf(1);
    for i=2:length(ISIpdf)
    ISIhazard(i)=ISIpdf(i)/(1-ISIcdf(i-1));
    end
    
% Set hazard rate of transitioning OUT of the ITI
% 1/65 is based on task parameters
    ITIhazard=(1/65);

    
% Generate sequence of observations that corresponds to trials
% Observations:
%   Null-->1
%   Odor ON-->2
%   Reward-->3
for i=1:nTrials
    if ~isnan(ISIdistributionMatrix(i)) %reward delivery trials
    ISI = ones(1,ISIdistributionMatrix(i));
    ITI = ones(1,geornd(ITIhazard));
    trial=[2;ISI';3;ITI'];
    x=[x;trial];
    else %omission trials
    ITI = ones(1,geornd(ITIhazard));
    trial=[2;ITI'];
    x=[x;trial];
    end
    ITIdistributionMatrix(i)=length(ITI);
end

% states:
% ISI = 1-14
% ITI = 15

%Fill out the observation matrix O
% O(x,y,:) = [a b c]
% a is the probability that observation 1 (null) was observed given that a
% transition from sub-state x-->y just occurred
% b is the probability that observation 2 (odor ON) was observed given that
% a transition from sub-state x-->y just occurred
% c is the probability that observation 2 (reward) was observed given that
% a transition from sub-state x-->y just occurred

O=zeros(15,15,3);

%ISI
    O(1,2,:) = [1 0 0];
    O(2,3,:) = [1 0 0];
    O(3,4,:) = [1 0 0];
    O(4,5,:) = [1 0 0];
    O(5,6,:) = [1 0 0];
    O(6,7,:) = [1 0 0];
    O(7,8,:) = [1 0 0];
    O(8,9,:) = [1 0 0];
    O(9,10,:) = [1 0 0];
    O(10,11,:) = [1 0 0];
    O(11,12,:) = [1 0 0];
    O(12,13,:) = [1 0 0];
    O(13,14,:) = [1 0 0];

%obtaining reward
    O(14,15,:) = [0 0 1];
    O(13,15,:) = [0 0 1];
    O(12,15,:) = [0 0 1];
    O(11,15,:) = [0 0 1];
    O(10,15,:) = [0 0 1];
    O(9,15,:) = [0 0 1];
    O(8,15,:) = [0 0 1];
    O(7,15,:) = [0 0 1];
    O(6,15,:) = [0 0 1];
    

%stimulus onset
    O(15,1,:) = [0 1 0]; %rewarded trial
    O(15,15,2) = ITIhazard*0.1; %omission trial
    
%ITI
    O(15,15,1) = 1-(ITIhazard*0.1);
    
    
%Fill out the transition matrix T
%T(x,y) is the probability of transitioning from sub-state x-->y
T=zeros(15,15);

%odor ON from substates 1-6
%no probability of transitioning out of ISI while odor ON
T(1,2)=1;
T(2,3)=1;
T(3,4)=1;
T(4,5)=1;
T(5,6)=1;

%T(ISIsubstate_i+6-->ISIsubstate_i+7) = ISIhazard(i)
%these substates span the variable ISI interval
%if reward is received, then transition into the ITI
for i=5:length(ISIhazard)+4
     T(1+i,2+i)=1-ISIhazard(i-4);
     T(1+i,15)=ISIhazard(i-4);
end
T(14,15)=1;

% ITI length is drawn from exponential distribution in task
% this is captured with single ITI substate with high self-transition
% probability
T(15,15)=1-(ITIhazard*0.9);
T(15,1)=ITIhazard*0.9;

%% Visualize the transition and observation matrices
%Code for Supplementary Figure 7
subplot(2,1,1)
imagesc(T)
title('Transition Matrix')
xlabel('Next Substate')
ylabel('Current Substate')

subplot(2,1,2)
imagesc(O)
title('Observation Matrix')
xlabel('Next Substate')
ylabel('Current Substate')
%% Run TD learning

results = TD(x,O,T);

%% plot RPE as a function of ISI; plot every trial

RewardIndices=find(x==3);
RewardIndices=RewardIndices(length(RewardIndices)*0.4:end); % only look at trials after 2000 trials
ISIdistributionMatrix_rewardedtrials=ISIdistributionMatrix(~isnan(ISIdistributionMatrix));
ISIsforplot=ISIdistributionMatrix_rewardedtrials(length(ISIdistributionMatrix_rewardedtrials)*0.4:end); % only look at trials after 2000 trials

plot(ISIsforplot,results.rpe(RewardIndices),'k*')
xlabel('ISI')
ylabel('TD error')
%% plot average RPE for each ISI
% Code for Supplementary Figure 5

RewardRPE=results.rpe(RewardIndices);

% Average RPEs (and standard error) for each ISI length
for i=1:9
    averageRPE(i)=sum(RewardRPE(find(ISIsforplot==i+4)))/length(find(ISIsforplot==i+4));
    errorRPE(i)=std(RewardRPE(find(ISIsforplot==i+4)))/sqrt(length(find(ISIsforplot==i+4)));
end

% Plotting average RPE and standard error for each ISI
for i=1:9
    errorbar(i, averageRPE(i), errorRPE(i),'k')
    hold on
    plot(i, averageRPE(i), '.','Color',[1-i*.1 i*.1 1],'markersize',25)
    hold on
end

xlabel('time of reward delivery','fontSize',20)
ylabel('Average TD error','fontSize',20)
%% Value, valueprime, and RPE
%Code for value signals and RPEs shown in Figure 6
clear cueonsets;
    cueonsets=find(x==2);
    whichISI=13; %how long is the ISI for the trial type that you want to plot the value signal for? range: 5-13
    cueonsets=cueonsets(ISIdistributionMatrix==whichISI);
    cueonsets=cueonsets(round(length(cueonsets)*0.4):end-12); %  only look at trials after 2000 trials
    
    value=zeros(1,20);
    valueprime=zeros(1,20);
    rpe=zeros(1,20);
    for i=1:length(cueonsets)
        for j=1:20
            value(j)=results.value(cueonsets(i)+j-2)+value(j);
            valueprime(j)=results.value(cueonsets(i)+j-1)+valueprime(j);
            rpe(j)=results.rpe(cueonsets(i)+j-2)+rpe(j);
        end
    end
    
% plot value, value(t+1) and rpe
    subplot(3,1,1)
    plot(value/length(cueonsets),'k')
    hold on
    plot(valueprime/length(cueonsets),'Color',[0.5 0.5 0.5])
    title('Value [black] and Value(t+1) [grey]')
    
    subplot(3,1,2)
    plot((valueprime-value)/length(cueonsets),'k')
    title('Value(t+1)-Value(t)')

    subplot(3,1,3)
    plot(rpe/length(cueonsets),'Color',[1-(whichISI-4)/9 (whichISI-4)/9 1])
    title('TD error')

%% Plot substate weights
%Code for weights shown in Figure 6
for i=1:15
    weight(i)=sum(results.w(round(length(results.w)*0.4):end,i));
end
bar(weight/(round(length(results.w)*0.6)))
ylim([0 2])
ylabel('Weight')
xlabel('substate')
%% Generate matrix of RPEs for plotting
%Code for RPEs shown in Figure 5
    RPE=zeros(10,30);
    allOdorIndices=find(x==2);
    odorIndicesforplot=allOdorIndices(length(allOdorIndices)*0.4:end-12);
    ISIsforplot=ISIdistributionMatrix(length(allOdorIndices)*0.4:end-12);
    
    for i=1:length(ISIsforplot)
        if ~isnan(ISIsforplot(i))
            RPE(ISIsforplot(i)-4,:)=(results.rpe((odorIndicesforplot(i)-5:odorIndicesforplot(i)+24)))'+RPE(ISIsforplot(i)-4,:);
        else
        end
        
    end

% divide by the number of trials to compute each ISI's average RPE
for i=1:size(RPE,1)
    RPE(i,:)=RPE(i,:)/sum(ISIsforplot==i+4);
end
plot((RPE(1:9,:))')
ylabel('TD Error')
xlabel('time')
%% Plot omission trials
% Code for Supplementary Figure 1
odorIndices=find(x==2);
Omissiontrials=odorIndices(isnan(ISIdistributionMatrix));
Omissiontrials=Omissiontrials(round(0.4*length(Omissiontrials)):end);
omissionRPEs=zeros(1,30);
for i=1:length(Omissiontrials)
    omissionRPEs(1:30)=omissionRPEs(1:30)+(results.rpe(Omissiontrials(i)-4:Omissiontrials(i)+25))';
end
omissionRPEs=omissionRPEs/length(Omissiontrials);
plot(-0.8:0.2:5,omissionRPEs)
ylim([-0.8 0.8])