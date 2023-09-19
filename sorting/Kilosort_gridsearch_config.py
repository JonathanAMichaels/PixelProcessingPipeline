from sklearn.model_selection import ParameterGrid


# Define the parameter grid to be searched over using ops variables from Kilosort_config_3.m
def get_KS_params_grid():
    grid = dict(
        Th=[[7, 3], [6, 3], [5, 3], [4, 3]],
        spkTh=[-6, -4, -2],
        # lam=[10],
        # nfilt_factor=[4],#[1,4,16],
        # AUCsplit=[0.8, 0.9], #,0.95,0.99],
        # momentum=[[20, 400], [60, 600]],
    )
    return ParameterGrid(grid)