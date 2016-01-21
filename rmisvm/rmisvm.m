function [w,b]=rmisvm(fbag,Y,options)
% where f is a set of bags of intances
% MIN_w lambda/2 |w|^2 + 1/N SUM_i LOSS(w, X(:,i), y(i))+beta*(-1)*SUM_i
% Yi*logPi+(1-Yi)log(1-Pi)
%  where LOSS(w,x,y) = MAX(0, 1 - y w'x) is the hinge loss while we replace y with p since y is not given in mil problem.

d=size(fbag{1,1},1);% feature dimension
N=size(fbag,2);
if(isfield (options,'lambda'))
    lambda=options.lambda;
else
    lambda=1/N;
end

if(isfield (options,'nbite'))
    nbite=options.nbite;
else
    nbite=100*N;
end

if(isfield (options,'gamma'))% constraining whether use instance-level loss
    gamma=options.gamma;
else
    gamma=1.0;
end

if(isfield (options,'m0'))
    m0=options.m0;
else
    m0=1;
end

if(isfield (options,'beta'))
    beta=options.beta;
else
    beta=1.0;
end

if(isfield (options,'biasmult'))
    biasmult=options.biasmult;
else
    biasmult=1.0;
end

W=zeros(d+(biasmult==1),1);

sqrtlambda=sqrt(lambda);
L=zeros(nbite,1);
for iter=0:1:nbite-1
    % randomly choose a bag from uniform distribution
    %         k=ceil(N*rand); %
    if mod(iter,N)==0
        sample = randperm(N);% good!!!
    end
    k=sample(mod(iter,N)+1);
    X=fbag{k};
    curbagla=Y(k);
    X=full(X);%NumDim*NumIns(in this bag)
    Xb=[X;ones(1,size(X,2))];% add up the bais term
    % calculate the Noisy-OR model
    acc=W'*Xb;% 1*NumIns
    pk=sigmoid(acc);
    Pk=1-prod(1-pk);
    % choose these instance in this bag that violate the hinge loss
    yk=sign(pk+eps-0.5);
    posflag=(yk.*acc<m0);% 1*NumIns
    % calculate the lost function
    [m,n]=size(Xb);
    epsilon=1.0/((iter+1)*lambda);
%     hingeloss=m0-yk.*acc;% 1*NumIns
%     L(iter+1)=1/2*lambda*(norm(W,2))+1.0*gamma/n*sum(hingeloss.*posflag)+beta*(-1)...
%         *(curbagla*log(Pk+eps)+(1-curbagla)*log(1-Pk+eps));
    %         % calculate the derivation and using gradient descent for optimization
    W=(1-lambda*epsilon)*W+1.0*gamma/n*epsilon*Xb*(yk.*posflag)'+beta*epsilon*Xb*...
        ((curbagla-Pk)/(Pk+eps)*pk');
    W=min(1.0,1/(sqrtlambda*norm(W,2)))*W;% gradient projection
end
% plot(L);
% pause;
W=min(1.0,1/(sqrtlambda*norm(W,2)))*W;
w=W(1:end-1);
b=W(end);

end
