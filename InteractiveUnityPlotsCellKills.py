#!/usr/bin/env python
# coding: utf-8

# # Purpose of this notebook:
# The purpose of this notebook is to plot an interactive aggregated data collected thus far from various Lingli cell kills.

# # Load Required Packages

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from collections import OrderedDict
from IPython.display import display
# Widget stuff
import ipywidgets as widgets
from ipywidgets import interact, interact_manual, fixed, interactive
from copy import deepcopy
from glob import glob


# # Define relevant functions

# In[21]:


def legend_without_duplicate_labels(ax, **kwargs):
    """Pass in an axis from matplotlib, will return a legend without duplicates"""
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), **kwargs)

def get_match_with_pre_data(data, experiment_condition, control_condition='Pre', field_to_search='Condition', unique_field='date', neuron='all'):
    """Given a dataframe with multiple conditions, return that single condition and matched control
    """
    # Make a copy so as to not modify entered data
    df = deepcopy(data)
    
    # Return only the requested neuron if asked
    if neuron != 'all':
        try:
            df = df[df.neuron==neuron]
        except AttributeError:
            df = df[df.Neuron==neuron]

    matched_data = []
    # Go through dataframe grouped by the relevant column, keep only relevant conditions
    for _, _data in df.groupby(unique_field):
        if control_condition not in _data[field_to_search].unique():
            continue
        if experiment_condition not in _data[field_to_search].unique():
            continue 
        keep = (experiment_condition, control_condition)
        matched_data.append(_data.loc[_data[field_to_search].isin(keep)])
    
    if len(matched_data) > 0:
        return pd.concat(matched_data) 
    
def split_match_data(matched_df):
    plot_vals = ['Burst Duration (sec)', '# of Spikes', 'Spike Frequency (Hz)',
                 'Instantaneous Period (sec)', 'Interburst Duration (s)', 'Duty Cycle']
    # Functions to apply over each condition
    funcs = [[np.median, np.mean, np.std]]*6
    # Create a dictionary to aggregate the grouped data by
    my_dict = dict(zip(plot_vals,funcs))
    try:
        agg_matched_df = matched_df.groupby(['Condition', 'date']).agg(my_dict).reset_index()
    except AttributeError:
        # If condition isn't in the dataframe
        return (None, None)
    pre_kill = agg_matched_df.loc[agg_matched_df['Condition']=='Pre']
    post_kill = agg_matched_df.loc[agg_matched_df['Condition']!='Pre']
    return pre_kill, post_kill


# # Load Data

# In[10]:


dataframe = pd.read_csv('lingli_aggkills_allneurons.csv')
neuron_names = dataframe.Neuron.unique()#  ['LG', 'LP', 'PY', 'DG', 'VD']
conditions =  ['2PD Kill', 'AB Kill', 'Dual Kill', '2LPG Kill']
dates = dataframe.date.unique()
y_values = ['Burst Duration (sec)', '# of Spikes', 'Spike Frequency (Hz)',
            'Instantaneous Period (sec)', 'Interburst Duration (s)', 'Duty Cycle']
markers = ['v', 'x', 'o', '*']
markers_d = dict(zip(conditions, markers))
cmap2=sns.diverging_palette(250, 125, l=60, n=len(conditions),sep=10, center="dark")
cmap_d = dict(zip(conditions, cmap2))
sns.set_context('talk')


# In[23]:


@widgets.interact(
    val=y_values,
    
    central_tendency = ['mean', 'median'],
    
    burst_num=(1, 150),
    
    neuron = neuron_names, 
    
    Condition = widgets.SelectMultiple(
        options=conditions,
        value=list(conditions),
        rows = len(conditions),
        description='Condition',
        disabled=False),
    
    dates=widgets.SelectMultiple(
        options=dates,
        value=tuple(dates),
        rows=5,
        description='Dates',
        disabled=False
        ))
def unity_plot(val='Burst Duration (sec)', central_tendency="mean",burst_num=10, 
                 neuron='LG', Condition=conditions, dates=dates):
    fig, ax = plt.subplots(ncols=1, figsize=(16,8))    
    if central_tendency.lower()=='mean':
        estimator = 'mean'
    elif central_tendency.lower()=='median':
        estimator = 'median'
   
    global dataframe
    global markers_d
    global cmap2
    df = dataframe.loc[(dataframe['date'].isin(dates)) & (dataframe['Burst#'] < burst_num)]
    #-----> Plot PD Kills
    for index, _condition in enumerate(Condition):
        match = get_match_with_pre_data(df, control_condition='Pre', experiment_condition=_condition, neuron=neuron)
        pre_kills, post_kills = split_match_data(match)
        if pre_kills is not None:
            pre_mean, exp_mean = pre_kills[val][estimator], post_kills[val][estimator] 
            ax.scatter(pre_mean, exp_mean, marker=markers_d[_condition], color=cmap_d[_condition], s=150, label=_condition)
            # Add in experimental and control error bars
            pre_std, exp_std = pre_kills[val]['std'], post_kills[val]['std'] 
            ax.errorbar(pre_mean, exp_mean, yerr=exp_std, capsize=10, capthick=3, zorder=1, color=cmap_d[_condition],label='', ls='none')
            ax.errorbar(pre_mean, exp_mean, xerr=pre_std, capsize=10, capthick=3, zorder=1, color=cmap_d[_condition],label='', ls='none')
    # Plot a unity line
    xlims = ax.get_xlim()
    x = np.linspace(*xlims)
    ax.plot(x, x, ls='--', lw=3, color='k', zorder=0)
    ax.set_xlim(xlims)
    # Make it look pretty
    ax.set_ylabel('{} (exp.)'.format(val), fontsize=20)
    ax.set_xlabel('{} (control)'.format(val), fontsize=20)
    
    lgd = legend_without_duplicate_labels(ax, bbox_to_anchor=(1.05, 1.015), markerscale=.8)
    title = '{} Data: Pacemaker Kill Experiments {}'
    ax.set_title(title.format(neuron, estimator), fontsize=20)   
    plt.show()

