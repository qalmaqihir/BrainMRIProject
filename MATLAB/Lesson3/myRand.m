function [a,s] = myRand(low,high)
a=low+rand(3,4)*(high-low);
v=a(:);
s = sum(v);
end
%%
% function [out_arg1, out_arg2, ...] = function_name(in_arg1, in_arg2, ...)
