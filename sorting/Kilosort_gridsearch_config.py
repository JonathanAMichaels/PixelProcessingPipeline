from sklearn.model_selection import ParameterGrid


# Define the parameter grid to be searched over using ops variables from Kilosort_config_3.m
# All parameter combinations are tried, with lower parameters being fully explored first
def get_KS_params_grid():
    grid = dict(
        Th=[[11, 8], [9, 6], [7, 4], [5, 2]],
        # spkTh=[[-2, -4, -6, -8], [-4, -6, -8], [-6, -8, -10]],
        lam=[10, 15],
        # nfilt_factor=[12, 4],  # [1,4,16],
        # AUCsplit=[0.8, 0.9], #,0.95,0.99],
        # momentum=[[20, 400], [60, 600]],
    )
    return ParameterGrid(grid)
