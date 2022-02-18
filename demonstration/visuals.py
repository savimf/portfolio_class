import seaborn as sns

sns.set(
    font='Serif',
    rc={
    'axes.axisbelow': False,
	'axes.edgecolor': 'lightgrey',
	'axes.facecolor': 'None',
	'axes.grid': False,
	'axes.labelcolor': 'black',
	'axes.spines.right': False,
	'axes.spines.top': False,
	'figure.facecolor': 'white',
	'lines.solid_capstyle': 'round',
	'patch.edgecolor': 'w',
	'patch.force_edgecolor': True,
	'text.color': 'black',
	'xtick.bottom': False,
	'xtick.color': 'black',
	'xtick.direction': 'out',
	'xtick.top': False,
	'ytick.color': 'black',
	'ytick.direction': 'out',
	'ytick.left': False,
	'ytick.right': False
    }
)

sns.set_context(
    'notebook',
    rc={
 	'font.size':16,
 	'axes.titlesize':20,
    'axes.labelsize':18
    }
)
