function QG = BC(ckey,time)

    alph=73*pi/180;
    QG = @(x,y,n) 1*(y<tan(alph)*x)+3*(y>=tan(alph)*x);
    
end