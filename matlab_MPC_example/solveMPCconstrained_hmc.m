function U = solveMPCconstrained_hmc(x0,u,xr,param)


% options = optimoptions('fmincon','SpecifyObjectiveGradient',true,'display','off');
% U = fmincon(@(x) newCost_hmc(x,x0,xr,param),u,[],[],[],[],[],[],@(x)newLinConstraintsHacky_hmc(x,x0,param),options);
% U = fmincon(@(x) newCost_hmc(x,x0,xr,param),u,[],[],[],[],[],[],[],options);

options = optimoptions('fminunc','SpecifyObjectiveGradient',false,'display','off','UseParallel',true);
U = fminunc(@(x) newCost_hmc_wLB(x,x0,xr,param),u,options);


end