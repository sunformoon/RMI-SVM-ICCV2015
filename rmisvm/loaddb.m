function [fbag, label2] = loaddb(p,norm)

if~exist('norm','var')
    norm=0;
end

load(p);

if norm==1||norm==2
%     X=full(features);
    features=lnorm(features,norm);
%     features=sparse(X);
end


idx = unique( bag_ids );
fbag = cell( 1, length(idx));
label2 = zeros( length(idx), 1 );

for i = 1:length( idx )
    fbag{i} = features( bag_ids == idx(i), : )';
    label2(i) = max( labels ( bag_ids == idx(i) ));
end

label2(label2<0) = 0;