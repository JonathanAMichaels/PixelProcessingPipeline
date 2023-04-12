# **ks25** major changes & updates

[ks25] is a heavily-modified version of the Kilosort spike sorting package for Matlab, tailored for the analysis of non-chronic linear array recordings with modest channel counts (~24-100 ch).

[My **\[ks25\]** branch](https://github.com/czuba/Kilosort) is based on the Kilosort 2.5 codebase, and attempts to marry some of the more successful features of Kilosort 2.0 (i.e. temporally dynamic waveform templates) with backported features & improvements from the newer Kilosort 2.5 & 3.0 (i.e. a modified version of the 'datashift2' drift correction algorithm).

While the original Kilosort package (by [Marius Pachitariu, et al.](http://github.com/MouseLand/Kilosort)) is an amazing resource for fast & accurate spike sorting of high-channel count high-density silicon probes (e.g. [Neuropixels](https://www.biorxiv.org/content/10.1101/2020.10.27.358291v1); ~100s-1,000s ch, ~15-20µm spacing), out-of-the-box results have been less successful with data collected during non-chronic linear array recordings with comparatively broader inter-electrode spacing (e.g. [Plexon U-Probes](https://plexon.com/products/plexon-u-probe/); 24-64 ch, ~50-100µm spacing).GUI updates

## Overview of changes

Using default Kilosort parameters --directly, or with modest adjustments-- to sort non-chronic recordings from 32-channel linear arrays fails to capture spikes that are easily detectible in the raw & filtered data traces (i.e. well above noise of continuous voltage raw data; see github issue [#63](https://github.com/MouseLand/Kilosort/issues/63) for example screenshot of missed high-amplitude spikes). These issues have been persistent since Kilosort 2.0, and remain throughout versions 2.5 & 3.0.

### Preventing template inversions

Amongst other updates & revisions, the **[ks25]** adaptation of Kilosort attempts to address two primary sources of missed units:

1. arbitrary inversion of low-rank template representations (consequence of template updating procedure; w/in template learning & extraction: `learnTemplates.m` & `trackAndSort.m`, respectively)
2. inadvertent cancellation of spike waveforms with balanced positive- & negative-going components (consequence of template alignment w/in `mexSVDsmall2.cu`)
   - fix for this necessitated abandoning the ambiguous nature of aligning spikes to _either peak or valley_ (issue [#221](https://github.com/MouseLand/Kilosort/issues/221))
   - *all templates* are now aligned to the _minimum_ value

Over batches & learning, these two errors result in spurrious batch-sized stuttering of spike detection (github issue [#60](https://github.com/MouseLand/Kilosort/issues/60), [#175](https://github.com/MouseLand/Kilosort/issues/175)), and errrant temporal shifts & polarity inversions that accumulate to missed and/or inadvertently dropped high-amplitude spike templates.

In the process, significant changes have been made to the raw file handling & batch processing (addresses bug reported in issue [#219](https://github.com/MouseLand/Kilosort/issues/219)), `datashift`  drift correction estimation (including fix described in issue [#394](https://github.com/MouseLand/Kilosort/issues/394) ), template learning procedure, spike extraction, calculation of individual spike `template` & `template_feature` amplitudes (i.e. amplitudes output from `mexMPnu8.cu`  & saved for manual curation w/ [Phy2](https://github.com/cortex-lab/phy)), and more.

### Tightened amplitude projections

[ks25] utilizes "_pcTight" variants of certain mex/cuda functions, which compute template & feature projections using a temporal window more tightly restricted around the spike timestamp.

- instead of using the whole waveform `t= 0:nt0` , these variants use samples ranging from `t = 6:nt0-15` when computing feature & template projection amplitudes
  - using the default value of `nt0 = 61` samples, this corresponds to a _waveform projection window 40 samples long_
    - ...at 40kHz, this translates to a 1 usec waveform projection, instead of 1.5 usec
  - the full `nt0` range of samples are still used for all _spike subtraction operations_

Narrowing of the projection window to exclude less informative samples in the tails of the waveform improves specificity of template & amplitude features without leaving tails in the raw data after subtraction.

### `temp_wh.dat` renamed

Since the `datashift` method of drift correction is applied _directly to the preprocessed data file_, `ops.fproc`, the nature of this copy of the raw data has transitioned from a relatively temporary/working copy to the actual instantiation the drift correction. When applied (i.e. `ops.nblocks`>0), attempting to map spike timestamps back to the original raw file will [more than likely] not actually correspond to the voltage(s) produced at the time of the spiking response. Therefore, It is now important that this preprocessed copy of the data be retained after sorting. Correspondingly, **the default filename for the proprocessed data file has been updated to `proDat_<saveDirName>.dat`** (formerly called `temp_wh.dat`; still created w/in `preprocessDataSub.m`).

In additon to reflecting the non-temporary nature of the preprocessed data file, this naming scheme also adds important sort origin information to the file.

---

Finally, it is worth noting that **[ks25] revisions have been implemented with _a primary emphasis on accurate extraction_, at the marginal expense of processing expediency**. When applied to recordings from ~32 channels with raw file sizes in the range of 10-20 GB, these tradeoffs are manageable & [in my hands] necessary for usable spike sorting results. Its quite possible that the balance of time-vs-accuracy tradeoff is tipped when applied to recordings from 100s of channels (e.g. from neuropixels). In such cases, users may already be achieving suitable results from the standard/main [Kilosort](https://github.com/MouseLand/Kilosort) (ver 3.0 at time of writing).

---



## Basic [ks25] usage

- Launch standard gui by executing  `>> kilosort` from the Matlab command window
- Select your **data file** & **output directory** for this kilosort session
  - If you choose your data file _first_, the output directory will automatically populate with the data file directory
  - ...because I generally house my converted `.dat` files for a given day in separate `./raw` directory & all kilosort output directories from that day in a `./KiloSort` directory, <u>*I prefer to select output directory first, then select my data file*</u>  
- Select probe layout file appropriate for your device
  - default variables & trace view should populate in the gui automatically after a probe file has been selected
  - Note: this selection must be done even if the probe dropdown already shows the probe file you intend to use
- Run **`ksGUI_updatePars.m`** to apply [your] advanced parameter settings to the current gui instance
  - ***similar to*** simply clicking on the "Set advanced options" button, this will create a `ks` variable in the base workspace comprising a handle to the kilosort gui object
  - all updates to the `ops` struct w/in `ksGUI_updatePars.m` will be applied to the `ks.ops` struct of the GUI object
  - ***in addition*** this function ensures that changes made to `ks.ops` are applied to the parameter fields of the GUI
- Click on the **"Preprocess"** button in the Kilosort GUI interface
  - this will run initial data preprocessing operations & initialize all important variables in the `ks.rez` struct
  - create a filtered copy of your data in `<saveDir>/proDat_<saveDirName>.dat` (...replacement naming convention for "temp_wh.dat")
  - compute & plot data shift estimates (...useful diagnostic, even if no drift correction is going to be applied)
- After reviewing the driftMaps produced during the preprocessing stages, if all looks good,
  click on the **"Sort & Save"** button to complete the actual sorting & saving stages





# Parameter updates & initialization function



## New processing flags

Lots of new flags to fine tune various sorting operations.

Definitions & recommendations from **`ksGUI_updatePars.m`**:

### General file handling & verbosity

```matlab
%% Custom fields or flags
% Parallel memmap raw data for whitening/loading
% - much faster data access than standard fread() methods
% - see preprocessDataSub.m, get_batch.m, & get_whitening_matrix_faster.m
ops.useMemMapping   = 1; % def = 1;

% Plot/update figures during spike sorting
ops.fig = 2; % 1==standard plots, 2=extra debug plots (more verbose, marginally slower)

% Git repo status & diff
ops.useGit = 1;

```

### Pre-processing

```matlab
% flag to round [data]shifts to nearest electrode spacing integer
% - ALWAYS use integerShifts for [y-axis] spacing >=50 um
ops.integerShifts = 1;

% Target batch for drift alignment
% - if  <1, will be batch nearest targBatch% of total batches
% - if >=1, will be direct index to batch#
% - default = 2/3;
ops.targBatch = [0.3];
```

### Template learning

```matlab
% Randomize batch order during learning
% - provides more stable/effective set of learned templates across entire file
ops.learnRand = 1;

% clip template updating to a minimum number of contributing spikes
% - helps prevent inversions (due to subtle/irregular noise being injected into dWU0 output of mexMPnu8.cu)
% 20 spike cutoff works well for 10 sec batch
ops.clipMin = 15;
ops.clipMinFit = .8;  % can survive clipping if median accounts for at least this much variance (ratio of vexp./amp)

```

### Post-processing

```matlab
% Apply detailed ccg analysis function to remove double-counted spike clusters
% - Orig from Bondy fork, but integrated into standard around kilosort 2(.5)
% - this is useful feature, but actually makes manual curation somewhat more challenging,
%   because strong ccg peak is informative for merge decisions
% - best left disabled for probes (even hopes that it would allow threshold cutoff to be less errorprone didn't work out)
ops.rmDuplicates    = 0;

% split clusters by:  1==template projections, 2==amplitudes, 0==don't split
ops.splitClustersBy = 0;  % 0 recommended (2 relatively safe & effective)
% amplitude splits seem reasonably trustworthy (...template splits suceptible to oddities of templates (e.g. inversions))

% standard cutoff can be overly aggressive
% - best left disabled for probes
ops.applyCutoff = 0;
```



## chanMap creation

Updated handling & creation of recording device channel map file ("chanMap")

- `createChannelMapFile.m` has been updated to a functional form:

  ```matlab
  % function cm = createChannelMapFile(nChan, ysp, xsp, chPerRow, varargin)
  %
  % Functional version of Kilosort channel map creation
  % - chanMap file will be saved in default kilosort configFiles directory
  %   as <cm.name>.mat    (...fallback to [pwd], asks to overwrite if file already exists)
  % - include or specify any additional channel map fields as additional PV pair input:  ...'parameter', value...)
  % - ks25 updated to retain any additional recording device info the user chooses to include in chanMap file
  % 
  % INPUTS:
  %   nChan   = total number of channels
  %   ysp     = y spacing [um]
  %   xsp     = x spacing [um]
  %   chPerRow    = number of channels per row
  %
  %   additional/optional PV pairs:
  %       'name'      (def = 'example<nChan>ch<fs/1000>k')
  %       'fs'        (def = 40000)
  %       'configFxn' (def = '')
  % 
  % OUTPUTS:
  %   cm  = channel map struct
  ```

- Channel numbering & spacing convention:
  - channel numbers increase from left-to-right, proximal-to-distal
  - y-zero at most distal channel
  - y-coord advances in ysp increments up the probe
    - such that y-zero coincides with tareing the probe depth
      at first detectable entry into brain
    - absolute channel depths are then:  probe depth - y-coord
  - x-coords are balanced on either side of x-zero

# Non-chronic linear array parameters

## Probe geometry

[ks25] was developed with the intention/hope of providing more usable sorting results for a broad range of multi-channel recording devices that exist with a modest channel count & electrode spacing in the 50-100 um range. Testing & development has only been done with a single style/geometry of Plexon U-Probe recordings in awake-behaving NHP.

- 32 channel Plexon U-Probes
- stereotrode configuration

  - two rows of 16 electrode sites
  - [50, 100] um spacing   ( [x, y];  um [within, between] stereo pairs)
- continuous voltages acquired at 40kHz resolution using Plexon OmniPlex system ("DigiAmp" version)

Since recording depth is tared when the most distal electrode (ch32) discernibly enters brain (dampening of noise & faintly detectable hash), the kilosort channel map is formatted as:

This channel map is included in the `./configFiles` directory as `uProbe32stereo40k.mat`, and can be easily [re]created with the updated channel map creation function:

```matlab
cm = createChannelMapFile(32, 100, 50, 2, 'name','uProbe32stereo40k','configFxn','ksGUI_updatePars')
```

- for other chanMap default files (including those from the original Kilosort repository), check the `./configFiles/unused` directory



## General processing & thresholds:

```matlab
% when Kilosort does CAR, is on batch time blocks, which mitigates risk of steps injected btwn demeaned segments
% -- NOTE: flag is "CAR", but uses common median referencing [CMR]
ops.CAR                 = 1;
ops.useStableMode       = 1;

ops.throw_out_channels = 0; % NO! confounds source identity; never throw out chans during sort
ops.minfr_goodchannels = 0; % minimum firing rate on a "good" channel (0 to skip); always disable, see above   (def=0.1)

% [minFR] prevents errant 'units' with just a few detected spikes from proliferating
ops.minFR = 0.02; 
% - clip truly useless templates, but don't drop less active ones (esp with randomized batch order during learning)
ops.minFR = ops.minFR / max([ops.learnRand*10,1]);

% threshold used when establishing baseline templates from raw data
ops.spkTh = -6;     % [def= -6]
ops.ThPre = 8;      % [def= 8]

% splitting a cluster at the end requires at least this much isolation for each sub-cluster (max = 1)
ops.AUCsplit = 0.9; %0.95; % ks2 default=0.9;  ks3 default=0.8;

% how important is the amplitude penalty (like in Kilosort1, 0 means not used, 10 is average, 50 is a lot)
ops.lam = 10;  % ks3 default is 20; previously 10...    (TBC: this has always been extremely cryptic)

% threshold on projections (like in Kilosort1, can be different for last pass like [10 4])  (def=[10,4])
ops.Th = [8 4]; %[10 4];  (TBC: [8 4] better for awake nhp, but still clipping)

```

## Batch duration:

```matlab
% Define batches in seconds of data, not abstract bit chunks
batchSec 		= 10;  % define batch number in seconds of data  (TBC: 10 seems good for 1-2 hr files and/or 32 channels)

% samples of symmetrical buffer for whitening and spike detection
% - implementation FINALLY FIXED in ks25!! (...was broken since kilosort 2.0)
ops.ntbuff      = ceil(1*ops.fs/64)*64;%  ceil(batchSec/4*ops.fs/64)*64;%  64; % 64*300; % (def=64) 

% buffer size in samples
ops.NT          = ceil(batchSec*ops.fs/32)*32; % convert to 32 count increments of samples

% sample from batches more sparsely (in certain circumstances/analyses)
batchSkips      = ceil(60/batchSec); % do high-level assessments at least once every minute of data
ops.nskip       = 1; %batchSkips;  % 1; % how many batches to skip for determining spike PCs
ops.nSkipCov    = 1; %batchSkips;  % 1; % compute whitening matrix from every N-th batch

```

## Drift correction:

**Note:**  While the `datashift` method of drift correction is a great improvement on the batch reordering (temporal) method employed in Kilosort version 2.0, it is still recommended to only use the estimated drift maps as a diagnostic tool.

- If there are moments of the session in which a relatively abrupt shift occurs, use the drift estimates & `ops.integerShifts = 1` to identify the time of abrupt shift, then separately sort those segments of the file & merge the sorted results post-hoc
  - Admittedly, this is easier said than done. There is a new utility function called **`rezMergeToPhy.m`** that will [attempt to] merge the [rez] structs from two separate Kilosort sessions into one output directory that can be loaded into Phy for manual curation & integration.
  
  - ***Update/correction***:  due to the way that Kilosort & Phy currently interact, merging two `rez` structs together into something that resembles a usable Phy output is significantly more complicated than one might hope/expect. In light of significant changes that are expected from a [long-in-the-works](https://github.com/cortex-lab/phy/issues/1040) refactoring of Phy (_as of Nov 2020, it was stated that it was ["not going to be ready until some time next year"](https://github.com/cortex-lab/phy/issues/1042)_), and the still non-functional state of feature projections in Kilosort 3 (see [issue 317](https://github.com/MouseLand/Kilosort/issues/317)), putting more work into this band-aid session merge does not seem like the best way to go   
  
    ​    --TBC, circa June 2021
- If drift apparent in the driftMap is more gradual, try allowing template dynamics to absorb variation due to probe shift

```matlab
ops.nblocks = 0; % 0==driftMap will be computed & plotted, but not applied
% - attempt to allow template temporal dynamics to accomodate for slow drift during spike extraction phase
% - if abrupt drift is apparent, consider try rigid drift correction (.nblocks=1)

ops.integerShifts = 1; % *ALWAYS* use integerShifts for probes with spacing >= 50 um)

% "spatial smoothness constant for registration"
% - (TBC: relationship to ops.sigmaMask is confusing, but I **think** I've deciphered/standardized it...see description below)
ops.sig = 20; % [def = 20]
% - cant be 0; used by standalone_detector.m & 0 will wipe out multi-scale gaussian spread of generic templates across channels
% So looks like this param (or the often hardcoded .sig param) is used when applying datashift drift corrections in increments smaller than [y] sampling of recording sites
% - this effectively blurrs shifted data traces into/across adjacent channels
% - maybe doing so flies with high res sampling of neuropixels, but abruptly jacks data quality/signal on more coarsely sampled devices (e.g. uprobes)
% - DONT expand, like we do for sigmaMask; maybe minimize this as much as possible (==10? ==5?)
% - if ops.integerShifts==1, ops.sig is automatically set ==1 w/in datashift2.m

```

## Spike detection:

```matlab
%% Stereo-probe specific adjustments (standard geom: 50um within, 100um between stereopairs)
% spatial constant in um for computing residual variance of spike     (def=30)
% - 50-70 better for 50/100um intra/inter-trode spacing; else effectively zero spread across channels
ops.sigmaMask = 70;

% use all chanels available for whitening
% - used to accept = inf; but began throwing error after Kilosort 2.5  (TODO: retest)
ops.whiteningRange = 32;

% Spike detection: .loc_range & .long_range are used isolate threshold peaks in:
%       [nSamples in time, nChannels];
% - BUT relevant uprobe 'channel' units are very different from neuropixel spacing
%   - nChannels==3 will include lateral & longitudinally adjacent channels of stereo probe
%   - nChannels==5 will include adjacent stereoPAIRS (**but b/c spacing asymmetry of channel indices, this can include channels up to 300 microns away...:-/ )
ops.loc_range   = [5, 2];   % def=[5, 4]
ops.long_range  = [30, 4];  % def=[30, 6]

% [.nTEMP] number of initial templates to extract from threshold crossings
% - These form the basis for any new templates added during learning & the initial PCA dimensions
% - If undefined, 6 is the usual number of templates, but more seems generally non-detrimental & likely helpful
ops.nTEMP = 12;

```

## Template learning:

```matlab
% Randomize batch order during learning
% - provides more stable/effective set of learned templates across entire file
ops.learnRand = 1;

% number of samples to average over (annealed from first to second value)     (def=[20,400])
ops.momentum = [60 600]; % should really take batch size into account, but try expanding first [40 400];

% minimum number of contributing spikes required for template updating
% - helps prevent inversions (due to subtle/irregular noise being injected into dWU0 output of mexMPnu8.cu)
ops.clipMin = 20; % [def = 0] % 15-20 spikes recommended for ~5-10 sec batches

% low firing-rate template can escape clipping if median accounts for at least this much variance (ratio of vexp./amp)
ops.clipMinFit = 0.8; % [def = 0.7] % 0.8 recommended for uprobe spacing

```

## Post-processing:

- _Post-processing flags are largely implemented w/in the GUI Spike Sort operations;  ksGUI.m >> runSpikesort(obj)_
- General recommendation is to disable most/all post-processing auto merge & cutoff operations
  - This allows you to actually see what the outputs of the sorting algorithm are producting, before any clipping or modification
  - The nature of a temporally evolving spike template means that a single snapshot of a template at the end of sorting may not accurately represent the shape or channel distribution of the template used at the time of spike extraction. Therefore metrics of cluster quality based on this final/singular template shape tend to be misleading & result in inaccurate merging, splitting, or discarding of spike clusters all together

```matlab
% Apply detailed ccg analysis function to remove double-counted spike clusters
% - best left disabled for probes; information is useful for identifying merges during manual curation
ops.rmDuplicates    = 0;

% split clusters by:  1==template projections, 2==amplitudes, 0==don't split
% - amplitude splits seem reasonably trustworthy
% - template splits suceptible to template oddities/ideosyncrasies (e.g. inversions)
ops.splitClustersBy = 0;

% standard cutoff can be overly aggressive
% - best left disabled for probes
ops.applyCutoff = 0;

```



---

---

# Detailed explanation of code changes & motivations 



## 	Preprocessing updates

Summary

### Correct & consistent implementation of batch buffers

Loading of batches standardized using fully updated `get_batch.m`
- capible of loading either from Memory Mapped ("memmapped") file, or with standard `fread` calls
- a memmapped object handle to pre-processed data file (`temp_wh.dat`) is stored in `ops.fprocmmf`
  - mmf format must follow that setup in `preprocessDataSub.m`;  .Data subfield convention is `.chXsamp` (awk, but informative)

#### Extensive re-write of preprocessed file creation & handling

The datashift2.m drift correction approach necessitated a change in Kilosort processing pipeline in which drift correction is applied _directly to the pre-processed `temp_wh.dat` file_

- although drift correction shifts are documented in `rez.dshift`, they are implemented not merely as offsets, but _by rewriting the preprocessed `temp_wh.dat` file itself_
  - because channel is intrinsic to the byte order of the raw data file, this remapping involves projection of the raw continuous voltage signals onto a upsampled channel representation based on expected gaussian spatial distribution parameter `ops.sig`
  - therefore, one cannot approximate the shifted data based only on `rez.dshift` & the original raw data
  - **the preprocessed `temp_wh.dat` file must be preserved with kilosort outputs in order to extract waveform shapes based on the sorted spike times**

- Furthermore, for sorted spike times to be consistent with original/raw data timestamps when using a non-zero `ops.tstart`  (i.e. `ops.trange(1)~=0`), the preprocessed file _**must include all data from `t=0` to `t=ops.tend`, regardless of `ops.tstart`**_
  - Non-zero `tstart` can be important in cases where excess data epochs outside of the experimental stimulus/behavior are present in the raw acquisition file (e.g. comprising pre-experiment probe movement and/or settling time) 
  - because neural activity during excess epochs can be very distinct from that during the experimental stimulus, including them in the sorted data tends to cause all kinds of problems in the sorted outputs

#### Changes to `temp_wh.dat` file content

- carefully include ***all timepoints*** from `t=0` to `t=tstart` while maintaining consistent batch start/stop timepoints relative to `ops.tstart` & `ops.tend`. 
- update batch loading throughout codebase to use  `get_batch.m` calls for correct handling of batch sample offset/range/buffers/etc





### Drift correction (`datashift2.m`)

Summary

-  drift estimates are now _always computed_, regardless of `ops` settings
  - shift estimates output in `rez.dshift` 

- when enabled (`ops.nblocks>=1`), data shifts are **_applied directly to the preprocessed data file_** (`temp_wh.dat`) 
- drift correction/estimation occurs in 'batch sized' temporal resolution (`ops.NT` samples per batch)



#### Drift correction params

##### `ops.nblocks`

- default = 1;
- type of drift correction 
  - 0 = none;   rigid drift estimates still computed, but no correction is applied to data 
  - 1 = rigid;    all channels shifted up/down 
  - 2+ = nonrigid
  - non-rigid only really relevant for mmmany channels or probe length is long relative to brain (i.e. rodents)

##### `ops.integerShifts`

- default = 0;
- flag to round [data]shifts to nearest electrode spacing integer

##### `ops.targBatch`

- default = 2/3;
- preselect initial target batch for drift alignment
  - if  < 1, will be batch nearest `targBatch*100` percentile of total batches
  - if >= 1, will be direct index to batch#
- If stability/responsivity varies significantly across recording duration, may be helpful to tune the initial target batch to the most informative/representative timepoint w/in file 



Updated target selection & weighting



## Template inversion prevention



### Template update weighting

**`ops.momentum`** controls weighted update of template shape relative to new spikes detected on current batch. Originally

```matlab
fexp = exp(newSpikeCount.*log(pm));
dWU = dWU.*fexp + (1-fexp).*dWU0;
```

Where `dWU` is the current low-rank template, `dWU0` is the low-rank representation of the mean of recently detected spikes, and `pm` is the uptdate weight of the current batch, determined by `ops.momentum` as: 

```matlab
pmi = exp(-1./linspace(ops.momentum(2), ops.momentum(2), nbatches));
pm = pmi(ibatch); 
```

Two apparent downsides to original Kilosort approach:

1. all templates are updated based on the same batch weighting parameter, regardless of the time since that template was added/created. Thus, template flexibility that permits initial template refinement/evolution does not get extended to newly created templates in later batches of template learning phase(s)
2. if a sufficiently high number of spikes are detected within a given batch, the mean of newly detected spikes can completely overwhelm the current template record. Particularly for longer batch durations (e.g. 5-10 sec), increasing `ops.momentum(2)` to preclude this effect, can produce excessive rigidity of template fits for neurons with lower mean firing rates and/or templates that are added late in the learning process.

#### Changes to `ops.momentum` implementation

- **`maxweighting`** parameter to _**impose an upper limit on template updates**_
  -  `maxweighting` can be different for learning & extraction phases
  -  e.g. `=0.95` during learning to allow for maximum flexibility, then decreased to `=0.80` during extraction to prevent excessive drift
- **`filterAge`** parameter tracks batches since each template's***** creation during template learning, then _**apply individual weighting parameter based on each template's 'age'**_
  (***** _'filter'_, _'template'_, & _'cluster'_ terminology are used relatively interchangeably throughout codebase. If there is some subtle intended distinction between the three -- i.e. designating pre- vs. post- learning or manual curation -- it has so far escaped me.)

```matlab
maxWeighting = 0.95; % never completely wipe out template history in one batch
...
% updating weighting based on filter age
pm = pmi(min(filterAge, end))';
fexp = exp(fexp.*log(pm));  % exponentiate with weighting
fexp = fexp.*maxWeighting + (1-maxWeighting); % limit update weighting to [maxWeighting]% 
dWU = dWU.*fexp + (1-fexp).*dWU0;
```





### Update clipping via spike count & fit

Spurious template inversions seem to occur during batches were very few spikes were detected. ...working hypothesis is that a DC shift in the averaged components from just a few spikes effectively poisons the running [weighted] record of template shape

We don't want to impose a particular waveform shape/polarity de novo, so constraining template shape during learning is not a good solution. Better approach is to exclude potentially noisy updates based on a minimum number of spikes detected during the current batch.

##### `ops.clipMin`

- default = 0;  no clipping

Impose a minimum number of spikes necessary for updating a template on any given batch.

- inherently a very batch-duration & spike-rate dependent parameter (...makes it tricky)

- at the expense of processing speed/efficiency, this can mitigated by setting the batch buffer length [ops.ntbuff] equal to a significant fraction of the actual batch length [ops.NT] (...even >= 50%)

##### `ops.clipMinFit`

- default = 0.7;

A second clipping theshold based on template fit quality to _exclude low firing rate candidates from clipping_ if detected spikes were sufficiently well accounted for by the spike template shape

- best I can discern, outputs of `mexMPnu8.cu`  for each detected spike define: 

  - `id0`==best fitting template index
  - `x0`== amplitude of threshold crossing of each spike candidate
  - `vexp` == magnitude of threshold amplitude variance explained by the spike template (??)

- each spike template fit can be approximated as  `vexp./x0`

- Thus, minimum spike count & fit quality thresholds for updating templates are determined by:

  ```matlab
  isclip = (newSpikeCount>0) & (newSpikeCount<clipMin);
  for ggg = find(isclip)'
      g = id0==ggg; % spikes from this template
      % only continue with clip if spikes detected are not very good fits
      isclip(ggg) = median(vexp(g)./x0(g)) <= clipMinFit;
  end
  newSpikeCount(isclip) = 0;
  ```





## Template learning

Summary

#### Rationale for random order batches during learning

If the goal of template learning is to produce _the complete set of templates_ necessary for spike extraction throughout the file, then fitting a set of templates to a temporally randomized batch order with a generally low[er] minimum spike rate threshold for dropping templates should produce the most robust template set.



### Multi-phase learning

#### Phase 1: Initial learning

- [iorder0] input to learnTemplates.m
  - length == [niter0]
- randomize order of batches from 2:end-1 for learning
  - skip first and last because have ideosyncratic edge artifacts (padded buffer segments), and tend to comprise non-stimulus epochs just before/after experimental stimulus
- template learning rate [ops.momentum] spans this range as normal;  `exp(-1./linspace(ops.momentum(1), ops.momentum(2), niter0))`
- template dropping & adding occurs as normal (every 5th batch) during learning phase 1:
  - templates are dropped (triageTemplates2.m) a during this phase based on standard metrics
    - min firing rate cutoff [ops.minFR] (needs batch length dependent update)
    - similarity to other templates  (would like to see this updated too; currently hardcoded cutoffs w/in triageTemplates2.m)
  - templates are added as detected in residuals
    - better understanding of this process (w/in `mexGetSpikes2.cu`) would be good
      - ...newly added templates seem to often include excessively broad and/or predominantly negative spatial (channel) weighting distributions

**NOTE:** length of Phase 2 & 3 dependent on [ops.hardeningBatches] parameter

  - can be manually set, but default 50% of total batch count set w/in learnTemplates.m is recommended
    (def= `ceil(0.50 * niter0)`)

#### Phase 2: Template hardening

- pad initial set of randomized batches with additional randomly ordered batches
- template dropping occurs as normal, _but_ template adding is disabled
- provides time for templates that may have been newly added during Phase 1 to undergo template shape evolution, and be dropped if they prove to be inactive (<minFR) or simply merge to become similar to existing template (i.e. standard triage drop crit)
- template update parameter [pmi] is held at final/highest value throughout this phase
  - **~!~** perhaps this param should be relative to the 'age' of each template rather than batch iteration #
    - _**~!~Done~!~**_ template update weighting is now indexed by `filterAge` parameter, which is updated across batches & throughout triage/drops/reordering operations during Learning Phases

#### Phase 3: Conditioning for start of extraction

- pad with sequence of batches from `[hardeningBatches]:-1:1`
- conditions templates by walking them back to first batch prior to advancing to spike extraction phase
- template dropping & adding are _both_ disabled during this phase



### New template generation

During 'phase 1'  of template learning (see Multi-phase learning section) new templates are queued based on clustered threshold crossings detected in the residuals of the current batch after spike extraction with the current set of templates (specifically `learnTemplates.m >> mexGetSpikes2.cu`).

- however, original Kilosort implementation does not actually initialize new templates based on the shape of residual spike clusters, rather it just initializes the temporal components with generic waveform PCA components (`rez.wPCA`) & ignores the spatial (channel weighting) components all together
  - the low-rank representation of the new template residuals (`dWU0` output from `mexGetSpikes2.cu`)  _are incorporated into the existing `dWU` low-rank templates_ 
  - and no doubt any missing components are computed during subsequent stages of processing on the next batch iteration (probably more efficiently on the gpu by `mexSVDsmall2.cu`)
  - _BUT_ generic PCA initialization surely skews new templates [somewhat] away from their motivating components, and effectively masks the shape what is driving new template additions (insofar as they're not apparent in the `make_fig.m`  plots)

### Refined template dropping

During 'phase 1 & 2' of template learning, templates are dropped based on:

- minimum firing rate (`ops.minFR`, def = 0.02)
- _**new:**_   drop any templates with predominantly negative weighting
  - based on either peak channel, or mean weight across all channels [in template]
- similarity to other templates





# Spike extraction

Summary

- during spike extraction the set of clusters produced from the Learning Phases remains fixed (none added, none removed)
- the template shape (temporal) & channel loading (spatial) components are updated to track [slow] changes in the spike shape & distribution across channels (drift) over time.

## Revived temporally dynamic waveform templates

Throughout the spike extraction process, cluster templates are updated based on SVD components of recently detected spikes, weighted by the number of spikes detected in the current batch.

- weighting magnitude is modulated by the final [ops.momentum] parameter (same as in Learning Phase)
- a record of template dynamics is stored in [rez.WA], [rez.UA], & [rez.muA]
  - rez.WA is sized:  `[nt0, Nfilt, Nrank, nBatches]`
  - rez.UA is sized:  `[Nchan, Nfilt, Nrank, nBatches]`
  - rez.muA is sized:  `[Nfilt, nBatches]`
  - ...rez.nspA also stores a rolling count of spikes detected per batch. This is [obv.] a bit redundant with spike times, but useful record to have & [comparatively] not outlandishly large.
- these fields can become rather large, but maintaining a record of template evolution is important for producing a meaningful amplitude scatter plot for use during manual curation



## Prevent spurious inversions of template components

### Align spike waveforms to minima

Traditionally Kilosort has taken an ambiguous approach to spike polarity (i.e. primarily 'negative-going' vs 'positive-going' voltage deviations) by aligning spikes to the `max(abs(samples))` . In theory, this is a totally reasonable -- and necessary-- approach for flexibility across different brain areas & neuron proximity to electrode. _But, in practice_ this causes destructive alignment of spikes with approximately balanced positive & negative extrema.

- incidences of this can be seen during template learning, where persistent bands of new spike templates appear on a given channel with excessively broad/low-amplitude templates and/or local maxima shifted significantly prior (<=sample 10) to the typical alignment point at `nt0min` (sample 20)

Although this potentially imposes a temporal offset for certain spike shapes, post-hoc adjustments to spike timeing alignment are a tractable problem, whereas wholly missed spike templates are not.

This update is wholly contained within `mexSVDsmall2.cu`, which is utilized to determine template shape during Learning & Extraction phases alike.



### Actively prevent inversions during extraction

Because the number & identity of each cluster is fixed during extraction, we can detect & prevent template inversions from occurring/propagating during extraction by comparing the current svd components to that of the previous batch (i.e. change in template before/after being updated by `mexSVDsmall2.cu` in `trackAndSort.m`)

- added step to compute dot product of temporal component of PC1 to that of previous batch
- the magnitude of template projections on each batch are displayed in a debug figure during extraction
  - set `ops.fig > 1`  to enable debug figure plotting 
- if projection is less than hard-coded threshold (`= 0.5` in trackAndSort.m) , then it is reset to previous state
- a record of  units & batches where inversions were detected is stored in rez.troubleUnits & rez.invDetected
  - if `ops.fig >= 1`, online visualization of inversions detected will be shown in the template projections debug figure  during spike extraction
  - examine inversions after sorting with:  `plotTemplateDynamics( rez, rez.troubleUnits);`

In practice, this does a pretty good job of preventing template-stuttering & maintaining cohesive/meaningful spike clusters throughout a file, without excessively limiting template dynamics or updating of low firing rate neurons. There tends to be a small number of templates that trigger this inversion from the get-go (i.e. as output from Learning Phases), and occasionally a template inversion will occur during extraction & remain persistent for an extended duration (part or all of remaining file). 

### Template update clipping

The same template updating clip parameters (`ops.clipMin` & `ops.clipMinFit`) used in Learning Phases are imposed during spike extraction.



---

---



# TODO items



## General

Lots of cryptic/unused parameter values in catchall [Params] input to cuda functions
- some of these may be b/c same [Params] set is passed into multiple functions, others just version detritus
  - _**notablly:**_  template update weighting parameter `pm` is refreshed as input `Params(9)` on each learning batch, but parameter is never used inside current cuda code.
    ...fortunate, because allows new `filterAge` parameter during learning without requiring edits to cuda source code 
- in any case, all undocumented & hard to parse



## GUI changes

In general, I've disabled most of the `try-catch` loops within gui execution. This can make things a little less pretty for the user, but they're way more debuggable when issues do arise. ....try-catch only when you must.

Updates to how probe files are handled w/in GUI internals to allow additional channel map fields (e.g. `.fs` for sampling frequency). This turned out to be somewhat more krufty than expected, b/c the gui actually loads ALL channel maps in the config dir into an indexed struct (which means _alll_ chanmap files would need to have the _exact same_ fields) and is surely why old `createValidChanMap.m` code was stripping away 'excess fields'.

- wrote some code around fixing this, and it seems to work, but likely to come up as a need for minor fix once others start using
- stumbled into strange initialization bug once channel map config files with different numbers of channels were used. ...if going from a higher to lower channel count config, the index into selected channels (which chans are displayed as traces) would consistently crash until reloaded the channel map and/or reselected the raw file. 
  - again, I think I patched this up sufficiently, but was just before pushing, so may still have some rough edges.





## Data preprocessing (.fproc)

Since the `temp_wh.dat` preprocessed file has morphed into a fullfleged copy of the data --onto which the datashift correction is applied-- restricting sort window to a time range with a non-zero start time causes ambiguity in the data-byte-to-time conversion that was previously implicit to the file structure.

- discrepancies between the timestamps of the original acquisition file to those of the raw or preprocessed data file are a problem waiting to happen
  -  if only converted a subset of the original file, then must append the correct offset when loading the sorted spiketimes in main analysis pipeline (likewise, that offset must be included whenever attempting to retrieve spike waveform samples)
  - Because the datashift drift correction is applied directly to the preprocessed data file (`temp_wh.dat`), restricting the preprocessing window inside kilosort also adds an unwise 'magic temporal offset' value.
- **Least bad solution** seems to be to ***always preprocess starting at* `t=0`**, then add offset consistent with `ops.tstart` value when loading batches
  - Causes larger `temp_wh.dat` file than necessary, but preserves correct spike timing without any nebulous offset parameters

### _Done!   (2021-05)_

All spikesorting stages (preprocessing, drift correction, template learning, & extraction) have been updated to reliably & consistently perform with any & all combination of non-zero **`ops.trange`** spike sorting ranges 

**the default filename for the proprocessed data file has been updated to `proDat_<saveDirName>.dat`** (formerly called `temp_wh.dat`; still created w/in `preprocessDataSub.m`).



## Drift correction (datashift2.m)

`[ops.integerShifts]` is effective in allowing datashift method of drift correction to be used on broader spaced linear arrays (without oblitterating the source amplitude)

- helpful when you _really do_ have abrupt shifts on the scale of electrode spacing (i.e. animal movement during epochs of unconstrained behavior between experimental stimulus files w/in a session)
- Less effective when underlying drift is primarily slow (i.e. didn't allow enough settling time after probe positioning)



Ideally, information from the driftMap & `.integerShifts` could be applied to automatically determine:

- sessions/epochs where temporal dynamics are likely to be sufficient to accomodate [slow] drift
- instances where `.integerShifts=1` are necessary to address abrupt shifts: 
  - define epochs of data where drift was relatively stable
  - apply independent sorts on stable epochs
  - stitch together templates based on integerShift magnitude & template correspondence at adjacent epoch ends
    - _ideally_ by seeding extraction with a "middle-out" approach w/in each epoch,  while 'memorizing' template state at either end   (i.e.  `midpoint:-1:1, memorizeStart, midpoint+1:end, memorizeEnd`)



Temporally dynamic updates to during extraction (`trackAndSort.m`) can be effective at tracking slow drift

- since template feature amplitudes are ***computed during extraction & written out to .npy during save*** (w/in `trackAndSort.m` & `rezToPhy.m` , respectively), _template feature projections **do** reflect temporal dynamics_ during spike extraction
  - additionally, narrowing the range of waveform samples onwhich spike feature amplitudes are computed significantly improved the usability of `template` & `template_feature` amplitude components during manual curation in Phy
    - previously spike amplitudes were computed on the entire set of `nt0` samples (typically 61)
    - refinements to `mexMPnu8_pcTight.cu` tightened the window of spike feature calculation to
      **`[6 : (nt0-15)]`**,  which with the default of nt0=61, results in projections on samples **`[6 : 45]`**
- BUT, template shape shown in Phy only reflects the state of the template at the end of extraction; there is [currently] no way to interrogate the timecourse of template shape w/in Phy
  - unclear what becomes of template & template_feature amplitude values after clusters are combined & split during manual curation...



## Template Learning

Rate parameters need updating to be adaptive to batch duration and/or nbatches
- [ops.momentum] (>> pmi >> pm) parameter currently [magic] weighting value based on number of spikes detected in current batch, but rate should be relative to batch duration
- [ops.minFR] minimum spike rate cutoff to drop templates is similarly coarsely implemented.
  - Cutoff is applied reasonably w/in triageTemplates2.m, but value to which it is applied is rolling weighted sum of spikes per batch as:   pastCount*[p1] + (1-p1)*newCount
  - default [p1] is 0.95; updated to 0.8 after setting up multi-phase learning with randomly order batches (goal to create set of templates suitable to entire file, not just most recent batches )



Now that newly added templates are *actually* based on the shape of residuals that produced them, worth revisiting template dropping based on similarity to existing templates

- if two templates evolve into a similar shape (as might happen as they descend into noise floor during probe drift), _but_ capture distinct spiking events/units at other timepoints, we don't want to completely/arbitrarily lose the one with lower amplitude at the time of triage assessment

