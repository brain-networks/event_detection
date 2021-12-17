function cinew = fcn_sort_communities(ci)
h = hist(ci,1:max(ci));
[~,idx] = sort(h,'descend');
cinew = zeros(size(ci));
for j = 1:max(ci)
    jdx = ci == idx(j);
    cinew(jdx) = j;
end