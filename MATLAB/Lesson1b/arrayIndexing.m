% Indexing Vectors

v = [1 3 4 6 9 12];
% transpose operator

c = v.';         % v row vector is tansposed to column vector
r= c.';          % column vector is transposed into row vector

v(1:4); % Indexing starts from 1 not 0

c;
c(2:5,1);

r(1:end);


% This code will start at 1, steps 2 till end
v;
v(1:2:end);
v(end:-2:1);

% n Linear spaced elements in btw a range (a,b)
% linspace(a,b,n)

s = linspace(0,100,10);

% Vector as index to another vector

t =  [2 4 6 8 10 12 14 16 18 20];
t([3 6 9]);


% ##########################################################%
%
% Indexing Matrix

% Seq of row vectors enclosed by [] & separated by ;
A = [1 2 3;5 6 7;9 10 11];  % 3x3 matrix

% Selecting the item at 2,1
A(2,1);

% Extracting a subarray 
t2 = A([1 3],[1 2 3]) % Extracts row 1 & 3, and columns 1,2,3
% t3 = A([rows],[cloumns])

% Same as above
t22 = A(1:3,1:3);

E = A([1 3],[3 2]);

% Extracting an entire column
C3= A(:,3);
% extracting entire row
R2= A(2,:);

% changing the values of martix using indexing
B=A;
B(:,3) = 0; % 3rd col got 0s

A(end,end);

A(end,end-2);
A;
A(2:end,end:-2:1); % good trick

% Single : select all the elements of a martix and arranges them into a
% column vector-> usefull in summing elements 
t2
v=t2(:)


% Sum the elements
% 1sth method call the sum func two times
A
colsSum = sum(A)
totalSum = sum(colsSum)
% these two steps are equal to sum(sum(A)) ;)

% 2nd method sum all the elements

goodSum = sum(A(:))


%########################################################%
%
% Logical Indexing
%

% Form A(D) A -> Array , D -> logical array

D = logical([1 0 1;0 0 1;0 0 0])
A =[1 2 3;4 5 6;7 8 9]
A(D) % gives a col vector of corresponding values 1s in A





