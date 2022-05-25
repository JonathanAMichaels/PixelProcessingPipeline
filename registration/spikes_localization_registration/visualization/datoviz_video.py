import numpy as np
from datoviz import canvas, run, colormap
import matplotlib.cm as cm
from tqdm import tqdm

fs = 30000 # (sampling rate)

### To plot only spikes with PTP > Threshold


#### Change the paths to where you stored the results arrays
threshold = 6
high_idx_first = np.load("max_ptp.npy")
high_idx_first = np.where(high_idx_first > threshold)[0]

x = np.load('x_results.npy')[high_idx_first]
y = np.load('y_results.npy')[high_idx_first]
z = np.load('z_results.npy')[high_idx_first]
clusters = np.load('yass_clusters.npy')[high_idx_first]
amp = np.load('max_ptp.npy')[high_idx_first]
st = np.load('spike_times.npy')[high_idx_first]/ fs
alpha = np.load('alpha.npy')[high_idx_first]

### Rescale alpha (brightness) between 0 and 100 for better visualization
alpha = np.log(alpha)
alpha = alpha - alpha.min()
alpha = (alpha/alpha.max())*100

pos0 = np.c_[x, z, y]
pos1 = np.c_[x, z, y]
pos2 = np.c_[x, z, amp]
pos3 = np.c_[x, z, alpha]

# Color.
log_ptp = amp
log_ptp[log_ptp >= 30] = 30
ptp_rescaled = (log_ptp - log_ptp.min())/(log_ptp.max() - log_ptp.min())
color = colormap(ptp_rescaled, alpha=5. / 255, cmap='spring')
color_clusters = colormap(clusters.astype('double')/clusters.max(), alpha=50. / 255, cmap='gist_ncar') 

# Create the visual.
c = canvas(width=800, height=1000, show_fps=True)
s = c.scene(cols=4)

p0 = s.panel(col=0, controller='arcball')
v0 = p0.visual('point', depth_test=True)
p1 = s.panel(col=1, controller='arcball')
v1 = p1.visual('point', depth_test=True)
p2 = s.panel(col=2, controller='arcball')
v2 = p2.visual('point', depth_test=True)
p3 = s.panel(col=3, controller='arcball')
v3 = p3.visual('point', depth_test=True)


p0.link_to(p1)
p0.link_to(p2)
p0.link_to(p3)


# Visual prop data.
v0.data('pos', pos0)
v0.data('color', color_clusters)
v0.data('ms', np.array([2]))

v1.data('pos', pos1)
v1.data('color', color)
v1.data('ms', np.array([2]))

v2.data('pos', pos2)
v2.data('color', color) #color_clusters
v2.data('ms', np.array([2]))

v3.data('pos', pos3)
v3.data('color', color)
v3.data('ms', np.array([2]))

# GUI with slider.
t = 0
dt = 20.0
gui = c.gui("GUI")
slider_offset = gui.control(
    "slider_float", "time offset", vmin=0, vmax=st[-1] - dt, value=0)
slider_dt = gui.control(
    "slider_float", "time interval", vmin=0.1, vmax=1000, value=dt)


def change_offset(value):
    assert value >= 0
    global t
    t = value
    i, j = np.searchsorted(st, [t, t + dt])
    color[:, 3] = 10
    color[i:j, 3] = 200
    color_clusters[:, 3] = 10
    color_clusters[i:j, 3] = 200
    v0.data('color', color_clusters)
    v1.data('color', color)
    v2.data('color', color)
    v3.data('color', color)

slider_offset.connect(change_offset)


@slider_dt.connect
def on_dt_changed(value):
    assert value >= 0
    global t, dt
    dt = value
    change_offset(t)


change_offset(0)
run()

