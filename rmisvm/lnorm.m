function norm_X=lnorm(X,p)
if p==0
    norm_X=X;% no norm
else
    % X numIns*Dim
    [m,n]=size(X);
    norm_X=zeros(size(X));
    for ii=1:m
        norm_X(ii,:)=X(ii,:)/(norm(X(ii,:),p)+eps);
    end
end
end