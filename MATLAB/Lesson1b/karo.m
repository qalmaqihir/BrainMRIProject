for i = 2:10
    for j = 2:10
        if(~mod(i,j))
            break;
        end
    end
    if (j > (i/j))
        fprintf('%d is  a prime\n', i);
    end
end