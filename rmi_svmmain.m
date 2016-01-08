addpath('./rmisvm/');
addpath('./data/MIL-Data-2002-Musk-Corel-Trec9-MATLAB/');

nfold  = 10;
ntime  = 10;

options.m0=0.2;
options.beta=6;
% options.nbite=1e4;
options.threshold=0.5;
options.lambda=0.019;
options.gamma=1;

% m0 beta lambda
% 0.8 7 0.005 for tiger dataset
% 1.5 5 0.017 for elephant dataset
% 0.2 6 0.019 for the fox dataset

[fbag, label] = loaddb('fox_100x100_matlab.mat',2);
fprintf('fox\n');

cvp = cell(1, ntime);
for n = 1:ntime
    cvp{n} = cvpartition( length(fbag), 'kfold', nfold );
end

finalacc=zeros(ntime,3);
for n = 1:ntime
    acc=zeros(nfold,3);
    for i = 1:nfold
        [w,b] = rmisvm( fbag(cvp{n}.training(i)), label(cvp{n}.training(i)) ,options);
        acc(i,:) =rmisvmpredict(fbag(cvp{n}.test(i)), label(cvp{n}.test(i)),w,b,options);
    end
    finalacc(n,:)=sum(acc,1)/nfold;
end
% average results
disp('average results:');
disp(mean(finalacc,1));
disp(std(finalacc,1));
% maximal results
[~,ind]=max(finalacc(:,1));
disp('best results:');
disp(finalacc(ind,:));