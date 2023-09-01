from sklearn.model_selection import ParameterGrid
# Define the parameter grid to be searched over using ops variables from Kilosort_config_3.m
def get_KS_params_grid():
    grid = dict(
        Th=[[7,3],[7,2]], #, [10,4], [9,4]],
        # lam=[10],
        # nfilt_factor=[4],#[1,4,16],
        # ntbuff=[64],#[512, 256, 128, 64],
        # AUCsplit=[0.8, 0.9], #,0.95,0.99],
        # momentum=[[20,400], [60,400], [120,400]],
        # spkTh=[-6,-4,-2],
    )
    return ParameterGrid(grid)