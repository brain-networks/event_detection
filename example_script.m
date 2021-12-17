clear all
close all
clc

%% detect events

load data/ts

% zscore time series
z = zscore(ts);

% number of time points/nodes
[t,n] = size(z);

% upper triangle indices (node pairs = edges)
[u,v] = find(triu(ones(n),1));

% edge time series
ets = z(:,u).*z(:,v);

% calculate rss
rss = sum(ets.^2,2).^0.5;

% repeat with randomized time series
numrand = 100;

% initialize array for null rss
rssr = zeros(t,numrand);

% perform numrand randomizations
for irand = 1:numrand
    
    % create circularly shifted time series
    zr = z;
    for i = 1:n
        zr(:,i) = circshift(zr(:,i),randi(t));
    end
    
    % edge time series with circshift data
    etsr = zr(:,u).*zr(:,v);
    
    % calcuate rss
    rssr(:,irand) = sum(etsr.^2,2).^0.5;
    
end

% calculate p-value
p = zeros(t,1);
for i = 1:t
    p(i) = mean(rssr(:) >= rss(i));
end

% apply statistical cutoff
pcrit = 0.001;

% find frames that pass statistical test
idx = find(p < pcrit);

% identify contiguous segments of frames that pass statistical test
dff = idx' - (1:length(idx));
unq = unique(dff);
nevents = length(unq);

% find the peak rss within each segment
idxpeak = zeros(nevents,1);
event_labels = idxpeak;
for ievent = 1:nevents
    idxevent = idx(dff == unq(ievent));
    rssevent = rss(idxevent);
    [~,idxmax] = max(rssevent);
    idxpeak(ievent) = idxevent(idxmax);
    event_labels(idxevent) = ievent;
end

%% cluster co-fluctuation time series

% get activity at peak
tspeaks = z(idxpeak,:);

% get edge indices
[u,v] = find(triu(ones(n),1));

% transform activity into co-fluctuations
ets = tspeaks(:,u).*tspeaks(:,v);

% calculate similarity matrix
rho = corr(ets');

% calculate mean similarity
mu = nanmean(rho(triu(ones(length(rho)),1) > 0));

% construct modularity matrix and symmetrize to remove imprecision
b = (rho - mu).*~eye(length(rho));
b = (b + b')/2;

% to run generalized louvain
% -download package here http://netwiki.amath.unc.edu/GenLouvain/GenLouvain
% -unzip contents
% -add whole directory to path with genpath(addpath(PATH_2_GENLOUV)) where 
%  PATH_2_GENLOUV is the location where the unzipped directory lives.

% number of louvain repetitions
nreps = 1000;

% run algorithm
ci = zeros(length(rho),nreps);
for irep = 1:nreps
    ci(:,irep) = genlouvain(b,[],false);
end

% run consensus clustering algorithm
cicon = consensus_communities(ci,nreps,true);

toc

%% predict fc

tic

% relabel communities, largest to smallest
cicon = sort_communities(cicon);

% calculate centroids
I = dummyvar(cicon);
I = bsxfun(@rdivide,I,sum(I));
cent = (I'*ets)';

% calculate and vectorize fc
fc = corr(ts);
fc = fc(triu(ones(n),1) > 0);

% calculate fraction of frames associated with each cluster
labels = event_labels;
labels(labels ~= 0) = cicon(labels(labels ~= 0));

numFrames = zeros(1,max(cicon));
for i = 1:max(cicon)
    numFrames(i) = sum(labels == i);
end

% number of clusters to use for predicting fc
numClu = 3;

% weight centroids by number of frames and calculate predicted fc
fcpred = nansum(bsxfun(@times,cent(:,1:numClu),numFrames(1:numClu))/sum(numFrames),2);

% compute correlation of predicted and observed fc
r = corr(fcpred,fc);

toc

%% plot rss time series, null, and significant peaks

f = figure('units','inches','position',[2,2,9,6]);
ax = axes('outerposition',[0,2/3,1,1/3]);
ph = plot(1:t,rssr,'color',ones(1,3)*0.65);
hold on;
qh = plot(idxpeak,rss(idxpeak),'r*',1:t,rss,'k','linewidth',2);
xlim([1,t])
xlabel('frame'); ylabel('rss');
legend([ph(1); qh],'null','significant','orig');

%% plot predicted FC

mu = fcpred;

% represent in matrix form
mat = zeros(n);
mat(triu(ones(n),1) > 0) = mu;
mat = mat + mat';

% load brain systems from Gordon et al
load data/hcp333
[~,idxsort] = sort(lab);

% draw matrix of co-fluctuation magnitude
axes('outerposition',[0,0,0.5,2/3]);
imagesc(mat(idxsort,idxsort),[-3,3]);
axis off

% add lines between systems
hold on;
idx = find(diff(lab(idxsort)));
for j = 1:length(idx)
    plot([0.5,n + 0.5],(idx(j) + 0.5)*ones(1,2),'k')
    plot((idx(j) + 0.5)*ones(1,2),[0.5,n + 0.5],'k')
end

% add system names
for i = 1:max(lab)
    x = mean(find(lab(idxsort) == i));
    text(-0.01*n,x,net{i},'horizontalalignment','right','fontsize',6)
    text(x,1.01*n,net{i},'horizontalalignment','right','fontsize',6,'rotation',45)
end

%% plot observed FC 

% calculate fc matrix
fcmat = corr(ts);

% draw matrix of static fc
axes('outerposition',[0.5,0,0.5,2/3]);
imagesc(fcmat(idxsort,idxsort).*~eye(n),[-1,1]);
axis off

% add lines between systems
hold on;
idx = find(diff(lab(idxsort)));
for j = 1:length(idx)
    plot([0.5,n + 0.5],(idx(j) + 0.5)*ones(1,2),'k')
    plot((idx(j) + 0.5)*ones(1,2),[0.5,n + 0.5],'k')
end

% add system names
for i = 1:max(lab)
    x = mean(find(lab(idxsort) == i));
    text(-0.01*n,x,net{i},'horizontalalignment','right','fontsize',6)
    text(x,1.01*n,net{i},'horizontalalignment','right','fontsize',6,'rotation',45)
end