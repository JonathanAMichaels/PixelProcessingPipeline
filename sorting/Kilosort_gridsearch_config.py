from sklearn.model_selection import ParameterGrid
# Define the parameter grid to be searched over using ops variables from Kilosort_config_3.m
def get_KS_params_grid():
    grid = dict(
        # Th=[[9,8], [7,5], [5,3]],
        lam=[5,10,20,40],
        nfilt_factor=[8,16,32],
        ThPre=[8,4,2],
        # AUCsplit=[0.9,0.95,0.99],
        # momentum=[[20,400], [60,400], [120,400]],
        # spkTh=[-6,-4,-2],
    )
    return ParameterGrid(grid)