function ciu = consensus_communities(ci,niter,vis)
% generate consensus communities
%
%   Inputs:
%
%       ci,     [node x iteration] matrix of partitions
%    niter,     number of times to cluster co-assignment matrix
%      vis,     true or false (default) for visualizing progress
%
%   Outpts:
%  
%      ciu,     consensus partitions
%
%   Requires Generalized Louvain function genlouvain.m
%   (http://netwiki.amath.unc.edu/GenLouvain/GenLouvain)
%
%   Requires Brain Connectivity Toolbox function (agreement) 
%   (https://sites.google.com/site/bctnet/)
%   
%   Rick Betzel, Indiana University, 2014
%

if ~exist('vis','var')
    vis = false;
end
N = size(ci,1);
mask = triu(ones(N),1) > 0;
if size(ci,1) == size(ci,2)
    d = ci;
else
    d = agreement(ci);
end
goFlag = length(unique(d));
if goFlag <= 2
    CiCon = ci;
end
while goFlag > 2
    
    mu = mean(d(mask));
    b = d - mu;
    b(1:(N + 1):end) = 0;
    CiCon = zeros(N,niter);
    for iRep = 1:niter
        CiCon(:,iRep) = genlouvain(b,10000,0);
        if vis & mod(iRep,10) == 0
            imagesc(CiCon); drawnow;
        end
    end
    d = agreement(CiCon);
    goFlag = length(unique(d));
    
end
ciu = CiCon(:,1);