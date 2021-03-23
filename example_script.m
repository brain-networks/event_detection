clear all
close all
clc

load ts

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
for ievent = 1:nevents
    idxevent = idx(dff == unq(ievent));
    rssevent = rss(idxevent);
    [~,idxmax] = max(rssevent);
    idxpeak(ievent) = idxevent(idxmax);
end

% get activity at peak
tspeaks = z(idxpeak,:);

% get co-fluctuation at peak
etspeaks = tspeaks(:,u).*tspeaks(:,v);

%% plot rss time series, null, and significant peaks
figure('position',[200,200,1400,200])
ph = plot(1:t,rssr,'color',ones(1,3)*0.65);
hold on;
qh = plot(idxpeak,rss(idxpeak),'r*',1:t,rss,'k','linewidth',2);
xlim([1,t])
xlabel('frame'); ylabel('rss');
legend([ph(1); qh],'null','significant','orig');

%% calculate mean co-fluctuation (edge time series) across all peaks
mu = nanmean(etspeaks,1);

% represent in matrix form
mat = zeros(n);
mat(triu(ones(n),1) > 0) = mu;
mat = mat + mat';

% load brain systems from Gordon et al
load hcp333
[~,idxsort] = sort(lab);

% draw matrix of co-fluctuation magnitude
figure('position',[200,200,600,600]);
axes('outerposition',[0.15,0.15,0.85,0.85]);
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
    text(-0.01*n,x,net{i},'horizontalalignment','right')
    text(x,1.01*n,net{i},'horizontalalignment','right','rotation',90)
end