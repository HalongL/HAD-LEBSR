function [A,S,J,L,W0,loss]=LEBSR(X,alpha,beta,m,mu,p,lambda,Dict,display)
epsilon=1e-7;
maxIter =500;
% maxIter =1e3;
% mu = 2;  
% m=10;
mu_max = 1e12; 
rou = 1.1;
[~,y]=size(Dict);
[dim,num] = size(X);
A=zeros(dim,m);
S=zeros(m,num);
% W0=ones(y,m);
W0=eye(y,m);
J=zeros(m,m);
L=zeros(m,m);
Z1=zeros(dim,m);
Z2=zeros(m,num);
Z3=zeros(m,num);
Y1=zeros(dim,m);
Y2=zeros(m,num);
Y3=zeros(m,num);
Y4=zeros(m,m);
I=eye(m,m);
Iy=eye(y,y);
Em=ones(m,1);
EN=ones(num,1);
DtD=(Dict'*Dict);
% DtD=eye(y,y);
% LtL=L*L';
% StS=S*S';
% AtA=A'*A;
% r=randi([1 num],1,1);
iter = 0;
loss=[];
while iter<maxIter
    iter = iter + 1;
    
    %updata S    
    temp=inv((A'*A)+(Em*Em')+2*mu*I)*(A'*X+Em*EN'+mu*Z2-Y2+mu*Z3-Y3);
%     temp=hyperNormalize(temp);
    S1=temp;
    
    %update A
    temp=(2*lambda*Dict*W0*L'+X*S1'+mu*(Z1-Y1/mu))*inv((S1*S1')+mu*I+2*lambda*(L*L)');
%     temp=hyperNormalize(temp);
    A1=temp;

   
    %updata Z1
    temp=max(A1+Y1/mu,0);
%     temp=hyperNormalize(temp);
    Z1=temp;
    
    %updata Z2
    temp=max(S1+Y2/mu,0);
%     temp=hyperNormalize(temp);
    Z2=temp;
    
    %updata Z3
    temp=S1+Y3/mu;
%     temp=hyperNormalize(temp);
    Z3=solve_Lp(temp,alpha/mu,p);
    
    
    % update J
    temp=L+Y4/mu;
    temp=svd_threshold(temp,beta/mu);
%     temp=hyperNormalize(temp);
    J1=temp;
  
%    updata L
    temp=inv(2*lambda*(A1'*A1)+mu*I)*(2*lambda*A1'*Dict*W0+mu*J1-Y4);
%     temp=hyperNormalize(temp);
    L1=temp;
    
    %updata W0
    temp=inv(2*lambda*(DtD+Iy))*(2*lambda*Dict'*A1*L1);
    temp=hyperNormalize(temp);
    W1=temp;

    Y1=Y1+mu*(A1-Z1);
    Y2=Y2+mu*(S1-Z2);
    Y3=Y3+mu*(S1-Z3);
    Y4=Y4+mu*(L1-J1);


%     updata mu
    mu=min(mu_max,rou*mu);
    
%       sc=norm(S1-S,'fro')/norm(S1,'fro');
    s1=norm(A1-Z1,Inf);
    s2=norm(S1-Z2,Inf);
    s3=norm(S1-Z3,Inf);
    s4=norm(L1-J1,inf);

    sm=[s1 s2 s3 s4];
    sc=max(sm);
    loss=[loss sc];
    A=A1;
    S=S1;
    J=J1;
    L=L1;
    W0=W1;

    
%     X=X1;

    if iter==1 || mod(iter,50)==0 || sc<epsilon
%     if display
        disp(['iter ' num2str(iter) ' ,mu=' num2str(mu,'%2.3e') ',stopC=' num2str(sc,'%2.3e')]);
    end
    if sc < epsilon
        break
    end
end


end