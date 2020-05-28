function U = solveMPCconstrained(x0,u,xr,param)


% options = optimoptions('fmincon','SpecifyObjectiveGradient',true,'display','off');
% U = fmincon(@(x) newCost(x,x0,xr,param),u,[],[],[],[],[],[],@(x)newLinConstraintsHacky(x,x0,param),options);

options = optimoptions('fminunc','SpecifyObjectiveGradient',false,'display','off','UseParallel',true);
U = fminunc(@(x) newCost_wlogBar(x,x0,xr,param),u,options);

end