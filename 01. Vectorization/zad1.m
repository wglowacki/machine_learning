%/*******************************************************/
%TASK 1
%vector of 10 elements
A = 2:2:20
A(2:2:end) %%even elements
V = A>10;
A(V) %%elements greater than 10
A(mod(A,4)==0)
sum(A(V)) %%sum of elements greater than 10

clear all; clc;
X=magic(5)
X(X>10) %%elements greater than 10
X(2:2:end,:)
X(:,1:2:end)
sum(sum(X))

%/*******************************************************/
%TASK 2
clear all; clc;
sum_n=0;
t0=clock;
for i=1:10000000
    sum_n = sum_n+i;
end
time_diff_loop = etime(clock, t0);

t1=clock;
A = 1:10000000;
sum(A);
time_diff_vec = etime(clock, t1);

%/*******************************************************/
%TASK 3
clear all; clc;
sum_n=0;
tic
for i=1:10000000
    sum_n = sum_n+1/(i*i);
end
toc

A = 1:10000000;
tic
sum(1./(A.*A));
toc

%/*******************************************************/
%TASK 4
clear all; clc;
sum_n = 0;
tic
for i=1:1000
    sum_n = sum_n+(i*i);
end
toc

A = 1:1000;
tic
sum((A.*A));
toc

%/*******************************************************/
%TASK 5
clear all; clc;
sum_n = 1;
sign = -1;
tic
for i=3:2:1003
    sum_n = sum_n + sign/i;
    sign = -sign;
end
toc

A = 3:2:1003;
tic
    a = sum(1./A(1:2:end));
    b = sum(1./A(2:2:end));
    result = 1-a+b
toc
%after vectorization time is greater

%/*******************************************************/
%TASK 6
clear all; clc;
X=[-1, 0, 2;
    1, -2, 1;
    -5, 0, 2];
X(X(1,:)>0)

%/*******************************************************/
%TASK 7
clear all; clc;
vector = 1:10
A = repmat(vector, 10, 1)

%/*******************************************************/
%TASK 8
clear all; clc;
vector = 1:10;
upper_trian = triu(repmat(vector', 1, 10))
down_train = tril(repmat(vector, 10, 1))
mydiag = diag(diag(upper_trian))
result = upper_trian + down_train - mydiag

%/*******************************************************/
%TASK 9
clear all; clc;
A = [0, 1, 1];
B = [1, 0, 1, 1];
bin2dec(num2str(A))
bin2dec(num2str(B))

%/*******************************************************/
%TASK 10
%a
clear all; clc;
A = [2, 1; 1, 2];
B = [3; -1];
result = inv(A)*B
%b
clear all; clc;
A = [2, 1, 3;
     3,-2, 4;
     1, 7,10];
B = [1, 17, 19]';
result = inv(A)*B