% Builds a spike matrix from a set of spike times...

function S = getSpkMat(Tlist,dt,T,isLogical)

if ~exist('T','var') || isempty(T)
    T=0;
    for i=1:length(Tlist)
        if max(Tlist{i})>T
            T=max(Tlist{i});
        end
    end
end

if iscell(Tlist)
    for i=1:length(Tlist)
%         fprintf('%03i/%03i ',i,length(Tlist));
        S(i,:) = getSpkVec(Tlist{i},dt,T,isLogical);
    end
else
    S = getSpkVec(Tlist,dt,T,isLogical);
end


function S = getSpkVec(tsp,dt,T,isLogical)

if nargin<4, isLogical=false; end

% tic

tsp = tsp(tsp>0);
tsp = tsp(tsp<T);

% fprintf('Building Spike Matrix...');

if ~isLogical
    S = sparse(zeros(1,ceil(T/dt)));
    % Allowing multiple spikes per bin...
    for j=1:length(tsp)
        S(ceil(tsp(j)/dt)) = S(ceil(tsp(j)/dt))+1;
    end
else
    S = (logical(zeros(1,ceil(T/dt))));
    % Max 1 spk/bin
    S(ceil(tsp/dt)) = 1;
    S = logical(S);
end

% toc