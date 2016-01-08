function re=rmisvmpredict(fbag,trY,w,b,options)

if(isfield (options,'threshold'))
    threshold=options.threshold;
else
    threshold=0.5;
end

d=size(fbag{1,1},1);% feature dimension
m=size(fbag,2);
preY=zeros(size(trY));
for kk=1:m
    X=fbag{kk};
    X=full(X);%NumDim*NumIns(in this bag)
    % calculate the Noisy-OR model
    acc=w'*X+b;% 1*NumIns
    pk=sigmoid(acc);
    Pk=1-prod(1-pk);
    if Pk>=threshold% threshold for Pij
        preY(kk)=1;
    end
end
re(1)=sum(preY==trY)/m;
re(2)=sum(preY.*(1-trY)/sum(1-trY));% false positive
re(3)=sum((1-preY).*trY/sum(trY));% false negative
end