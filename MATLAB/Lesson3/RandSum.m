function [a,s] = myRand(low,high)
a=low+rand(3,4)*(high-low);
v=a(:);
s = sumAllElements(a);
end


function summa  = sumAllElements(M)
global v;
v =M(:);
summa = sum(v);
end
