function rez = datashift2(rez, do_correction)


ops = rez.ops;
dispmap = ops.dispmap;


% The min and max of the y and x ranges of the channels
ymin = min(rez.yc);
ymax = max(rez.yc);
xmin = min(rez.xc);
xmax = max(rez.xc);

dmin = median(diff(unique(rez.yc)));
fprintf('vertical pitch size is %d \n', dmin)
rez.ops.dmin = dmin;
rez.ops.yup = ymin:dmin/2:ymax; % centers of the upsampled y positions

% dminx = median(diff(unique(rez.xc)));
yunq = unique(rez.yc);
mxc = zeros(numel(yunq), 1);
for j = 1:numel(yunq)
    xc = rez.xc(rez.yc==yunq(j));
    if numel(xc)>1
       mxc(j) = median(diff(sort(xc))); 
    end
end
dminx = max(5, median(mxc));
fprintf('horizontal pitch size is %d \n', dminx)

rez.ops.dminx = dminx;
nx = round((xmax-xmin) / (dminx/2)) + 1;
rez.ops.xup = linspace(xmin, xmax, nx); % centers of the upsampled x positions
disp(rez.ops.xup) 


if  getOr(rez.ops, 'nblocks', 1)==0
    rez.iorig = 1:rez.temp.Nbatch;
    return;
end

Nbatches      = rez.temp.Nbatch;


% index the shift at the channel locations
dispmap = dispmap(:, rez.yc+1);
% flip the dispmap
dispmap = -dispmap;

% grab the right time
NT = ops.NT;
ntb = ops.ntbuff;
batchstart = 0:NT:NT*Nbatches; % batches start at these timepoints
batchcentersamples = batchstart + floor((NT+ntb)/2);
batchcenterseconds = round(batchcentersamples / ops.fs);

batchcenterseconds(end) = batchcenterseconds(end-1); % the registration cuts off the end a bit
batchcenterseconds

dispmap = dispmap(batchcenterseconds,:);

% filter out any weird temporal things
[b,a] = butter(4,0.1);
dispmap = filtfilt(b, a, dispmap);

figure(1)
clf
imagesc(dispmap')
colorbar
print([ops.saveFolder 'displacement.png'], '-dpng');

if do_correction
    % sigma for the Gaussian process smoothing
    sig = rez.ops.sig;
    % register the data batch by batch
    dprev = gpuArray.zeros(ops.ntbuff,ops.Nchan, 'single');
    for ibatch = 1:Nbatches
        dprev = shift_batch_on_disk2(rez, ibatch, dispmap(ibatch, :), sig, dprev);
    end
    fprintf('time %2.2f, Shifted up/down %d batches. \n', toc, Nbatches)
else
    fprintf('time %2.2f, Skipped shifting %d batches. \n', toc, Nbatches)
end

rez.dshift = dispmap;

% next, we can just run a normal spike sorter, like Kilosort1, and forget about the transformation that has happened in here 

%%



