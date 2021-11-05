% Find local minima
minIndices = islocalmin(S,"MinProminence",80);

% Display results
clf
plot(S,"Color",[77 190 238]/255,"DisplayName","Input data")
hold on

% Plot local minima
plot(find(minIndices),S(minIndices),"v","Color",[237 177 32]/255,...
    "MarkerFaceColor",[237 177 32]/255,"DisplayName","Local minima")
title("Number of extrema: " + nnz(minIndices))
hold off
legend