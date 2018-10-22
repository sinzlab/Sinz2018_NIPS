import seaborn as sns
import numpy as np

movie_vs_noise_cmap = sns.xkcd_palette(['golden rod', 'cerulean'])
scan_order = ['17358-5-3', '17797-8-5' , '18142-6-3']
scan_cmap = sns.color_palette("husl", 8)[-3:]

#performance_yticks = np.linspace(0,.5,6)
#performance_ylim = (-.1, .6)

performance_yticks = np.linspace(0,.2,5)
performance_ylim = (0, .23)

def fix_axis(a):
    a.spines['bottom'].set_linewidth(1)
    a.spines['left'].set_linewidth(1)
    a.tick_params(axis='y', length=3)
    a.tick_params(axis='x', length=3)
    
def strike(text):
    result = ''
    for c in text:
        result = result + c + '\u0336'
    return result