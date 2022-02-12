function taxi_fare(d,t)
faire= 5+2*ceil(d) + ceil(t);
fprintf("A %.2fkm with %.2f minutes of waiting costs %d$\n",d,t, faire);
end
