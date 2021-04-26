from ipywidgets.widgets import Label, FloatProgress, FloatSlider, Button, Checkbox,FloatRangeSlider, Button, Text,FloatText,\
Dropdown,SelectMultiple, Layout, HBox, VBox, interactive, interact, Output,jslink
from IPython.display import display, clear_output
from ipywidgets import GridspecLayout
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
import ipywidgets

import time
import threading
import logging
import math
import numpy as np
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import pickle
import lmfit as lm
from lmfit.model import load_model, load_modelresult

import sys

import xps_peakfit
import xps_peakfit.models.models
from xps_peakfit import bkgrds as background_sub
from xps_peakfit.helper_functions import *
from xps_peakfit.gui_element_dicts import *
import xps_peakfit.autofit.autofit

from xps_peakfit.spectra import spectra
from xps_peakfit.sample import sample
# import xps_peakfit.auto_fitting
import os
import glob


class guiSample(sample):

    def __init__(self,dataload_obj = None ,bkgrd_subtraction_dict = None, data_dict_idx = None, overview = True, sputter_time = None, offval=0, plotflag = True, plotspan = False,\
        plot_legend = None,normalize_subtraction = False,name = None,spectra_colors = None,load_derk = False, **kws):

        super().__init__(dataload_obj = dataload_obj ,bkgrd_subtraction_dict = bkgrd_subtraction_dict, data_dict_idx = data_dict_idx, \
            overview = True, sputter_time = sputter_time, offval=offval, plotflag = False , plotspan = plotspan,\
        plot_legend =  plot_legend,normalize_subtraction = normalize_subtraction,name = name, spectra_colors = spectra_colors,load_derk = load_derk, **kws)



        save_figs_button = Button(description="Save Figures") 

        saved_root_name = Text(
            value=self.sample_name,
            placeholder='Save filename',
            disabled=False,
            layout = Layout(width = '200px', margin = '0 5px 0 0')
            )
        save_figs_chkbxs = {init_fig: Checkbox(
            value= False,
            description=str(init_fig),
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            ) for init_fig in ['Raw','Subtracted','Atomic_Percent'] }

        display( VBox( [saved_root_name,HBox( [HBox( list( save_figs_chkbxs[chks] for chks in save_figs_chkbxs.keys() ) ), save_figs_button] ) ] ) )
        # out = Output()
        # display(out)

        fig_dict = {}
        ax_dict = {}
        fig_dict['Raw'], ax_dict['Raw']= self.plot_all_spectra(offval = self.offval, plotspan = self.plotspan, saveflag=0,filepath = '',figdim=(15,10))
        fig_dict['Subtracted'], ax_dict['Subtracted'] = self.plot_all_sub(offval = self.offval)
        fig_dict['Atomic_Percent'], ax_dict['Atomic_Percent'] = self.plot_atomic_percent()

        @save_figs_button.on_click
        def save_figs_on_click(b):

            if not os.path.exists(os.path.join(os.getcwd(),'figures')):
                os.makedirs(os.path.join(os.getcwd(),'figures'))

            for figure in save_figs_chkbxs.keys():
                if save_figs_chkbxs[figure].value:

                    save_location = os.path.join( os.getcwd(),'figures',saved_root_name.value  + '_' + str(figure) )     
                    fig_dict[figure].savefig(save_location, bbox_inches='tight')

