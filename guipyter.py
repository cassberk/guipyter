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

import XPyS
import XPyS.models
from XPyS import bkgrds as background_sub
from XPyS.helper_functions import *
from XPyS.gui_element_dicts import *
import XPyS.autofit.autofit

from XPyS.spectra import spectra
# import XPyS.auto_fitting
import os
import glob



class ParameterWidgetGroup:
    """Modified from existing lmfit.ui.ipy_fitter"""
    def __init__(self, par, slider_ctrl=True,sliderlims = None):
        self.par = par
        self.slider_ctrl = slider_ctrl
        self.sliderlims = sliderlims

        widgetlayout = {'flex': '1 1 auto', 'width': 'auto', 'margin': '0px 0px 0px 0px'}
        width = {'description_width': '10px'}


        # Define widgets.
        self.value_text = FloatText(
            value=np.round(self.par.value,2),
            placeholder='Value',
            disabled=False,
            layout = Layout(width = '200px', margin = '0 5px 0 0')
            )
        self.expr_text = Text(
            value=self.par.expr,
            placeholder='Choose Your Destiny',
            disabled=False,
            layout = Layout(width = '200px', margin = '0 5px 0 0')
            )
        self.min_text = FloatText(
            value=np.round(self.par.min,2),
            placeholder='min',
            disabled=False,
            layout = Layout(width = '100px', margin = '0 0 15px 0')
            )
        self.max_text = FloatText(
            value=np.round(self.par.max,2),
            placeholder='min',
            disabled=False,
            layout = Layout(width = '100px', margin = '0 0 15px 0')
            )
        self.min_checkbox = Checkbox(description='min', style=width, layout=widgetlayout)
        self.max_checkbox = Checkbox(description='max', style=width, layout=widgetlayout)
        self.vary_checkbox = Checkbox(value=bool(self.par.vary),
            description=self.par.name,
            disabled=False,
            indent=False,
            layout = Layout(width = '200px', margin = '0 5px 0 0')
            )
        if self.slider_ctrl is True:
            # if self.par.expr is None:
            #     dis = False
            # else:
            #     dis = True
            self.ctrl_slider = FloatSlider (
                        value=self.par.value,
                        min = self.sliderlims[0],
                        max = self.sliderlims[1], ### Need to figure out a way to set this
                        step  = 0.01,
                        # disabled=dis,
                        description = self.par.name,
                        style = {'description_width': 'initial','handle_color' : element_color['_'.join(self.par.name.split('_')[:-1]+[''])]},
                        layout = Layout(width = '350px', margin = '0 0 5ps 0')
                        )

            widget_link =jslink((self.ctrl_slider, 'value'), (self.value_text, 'value'))
        # else:
        #     self.ctrl_slider = None


        # Set widget values and visibility.
        if self.par.value is not None:
            self.value_text.value = self.par.value
        min_unset = self.par.min is None or self.par.min == -np.inf
        max_unset = self.par.max is None or self.par.max == np.inf
        self.min_checkbox.value = not min_unset
        self.min_text.value = self.par.min
        # self.min_text.disabled = min_unset
        self.max_checkbox.value = not max_unset
        self.max_text.value = self.par.max
        # self.max_text.disabled = max_unset
        self.vary_checkbox.value = bool(self.par.vary)

        # Configure widgets to sync with par attributes.
        self.value_text.observe(self._on_value_change, names='value')
        self.expr_text.observe(self._on_expr_change, names='value')

        self.min_text.observe(self._on_min_value_change, names='value')
        self.max_text.observe(self._on_max_value_change, names='value')
        # self.min_checkbox.observe(self._on_min_checkbox_change, names='value')
        # self.max_checkbox.observe(self._on_max_checkbox_change, names='value')
        self.vary_checkbox.observe(self._on_vary_change, names='value')

    def _on_value_change(self, change):
        self.par.set(value=change['new'])

    def _on_expr_change(self, change):
        self.par.set(expr=change['new'])

    def _on_min_checkbox_change(self, change):
        self.min_text.disabled = not change['new']
        if not change['new']:
            self.min_text.value = -np.inf

    def _on_max_checkbox_change(self, change):
        self.max_text.disabled = not change['new']
        if not change['new']:
            self.max_text.value = np.inf

    def _on_min_value_change(self, change):
        if not self.min_checkbox.disabled:
            self.par.set(min=change['new'])

    def _on_max_value_change(self, change):
        if not self.max_checkbox.disabled:
            self.par.set(max=change['new'])

    def _on_vary_change(self, change):
        self.par.set(vary=change['new'])

    def close(self):
        # one convenience method to close (i.e., hide and disconnect) all
        # widgets in this group
        self.value_text.close()
        self.expr_text.close()
        self.min_text.close()
        self.max_text.close()
        self.vary_checkbox.close()
        self.min_checkbox.close()
        self.max_checkbox.close()

    def get_widget(self):
        box = VBox([self.vary_checkbox, 
                            self.value_text, 
                            self.expr_text, 
                            HBox([self.min_text, self.max_text]),
                            ])
        return box

    def update_widget_group(self,par):
        # print(par)
        self.par = par
        self.vary_checkbox.value =  bool(self.par.vary)
        self.min_text.value =  self.par.min
        self.max_text.value =  self.par.max
        # self.define_widgets()
        if par.expr != None:
            self.expr_text.value = self.par.expr
        elif self.par.expr == None:
            self.value_text.value = self.par.value


        # print(self.expr_text.value)
        

    # Make it easy to set the widget attributes directly.
    @property
    def value(self):
        return self.value_text.value

    @value.setter
    def value(self, value):
        self.value_text.value = value

    @property
    def expr(self):
        return self.expr_text.value

    @value.setter
    def expr(self, expr):
        self.expr_text.value = expr

    @property
    def vary(self):
        return self.vary_checkbox.value

    @vary.setter
    def vary(self, value):
        self.vary_checkbox.value = value

    @property
    def min(self):
        return self.min_text.value

    @min.setter
    def min(self, value):
        self.min_text.value = value

    @property
    def max(self):
        return self.max_text.value

    @max.setter
    def max(self, value):
        self.max_text.value = value

    @property
    def name(self):
        return self.par.name


class fitting_panel:

    def __init__(self, fit_object, n_scans):
        self.fit_object = fit_object
       
    # def interactive_fit(self):
 

        """Fitting functions supported by lmfit"""
        fitting_options = [('Levenberg-Marquardt', 'leastsq'),\
        ('Least-Squares minimization Trust Region Reflective method ',' least_squares'),\
        ('differential evolution','differential_evolution'),\
        ('brute force method', 'brute'),\
        ('basinhopping', 'basinhopping'),\
        ('Adaptive Memory Programming for Global Optimization','ampgo'),\
        ('Nelder-Mead','nelder'),\
        ('L-BFGS-B','lbfgsb'),\
        ('Powell','powell'),\
        ('Conjugate-Gradient','cg'),\
        ('Newton-CG','newton'),\
        ('Cobyla','cobyla'),\
        ('BFGS','bfgs'),\
        ('Truncated Newton','tnc'),\
        ('Newton-CG trust-region','trust-ncg'),\
        ('nearly exact trust-region','trust-exact'),\
        ('Newton GLTR trust-region','trust-krylov'),\
        ('trust-region for constrained optimization','trust-constr'),\
        ('Dog-leg trust-region','dogleg'),\
        ('Sequential Linear Squares Programming','slsqp'),\
        ('Maximum likelihood via Monte-Carlo Markov Chain','emcee'),\
        ('Simplicial Homology Global Optimization','shgo'),\
        ('Dual Annealing optimization','dual_annealing')]


        self.fit_button = Button(description="Fit")
        self.plot_button = Button(description="Plot")   
        self.save_params_button = Button(description="Save Parameters")      
        self.save_fig_button = Button(description="Save Figure") 
        self.autofit_button = Button(description="View Autofit")

          
        
        self.fit_method_widget = Dropdown(
            options = fitting_options,
            value='powell',
            description = 'Fit Method',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '400px', margin = '0 0 5ps 0')
            )
          
        self.BE_adjust_ref_widget = FloatText(
#             value=adjust_reference,
            description = 'BE Peak Ref.',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '400px', margin = '0 0 5ps 0')
            )
            
        self.spectra_to_fit_widget = SelectMultiple(
            # options=[None, 'All'] + list(np.arange(0,len(self.spectra_object.data['isub']))), # Get rid ofthis once data dict is no longer used
            options=[None, 'All'] + list(np.arange(0,n_scans)), # Get rid ofthis once data dict is no longer used
            value = ('All',),
            description='Spectra to fit',
            style = {'description_width': 'initial'},
            disabled=False
            )
        
        self.plot_all_chkbx = Checkbox(
            value= False,
            description='Plot all fit_results',
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            )
        
        self.ref_lines_chkbx = Checkbox(
            value= True,
            description='Plot reference lines',
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            )

        self.use_prev_fit_result_params = Checkbox(
            value= False,
            description='Update Params with Previous Fit Result',
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            )     
        
        self.plot_with_background_sub = Checkbox(
            value= False,
            description='Plot on background',
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            )

        
        self.save_params_name = Text(
            # value=self.spectra_object.spectra_name+'_fitparams',
            description = 'Save parameter name',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '400px', margin = '0 0 5ps 0')
            )
 
        self.save_fig_name = Text(
            # value=self.spectra_object.spectra_name,
            description = 'Save figure name',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '400px', margin = '0 0 5ps 0')
            )

        self.autofit_chkbx= Checkbox(
            value= False,
            description='autofit',
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            )     

        self.plotfit_chkbx= Checkbox(
            value= True,
            description='plot fits',
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            )      

        self.select_parameters_widget = Dropdown(
            # options = [None] + list(np.arange(0,len(self.spectra_object.data['isub']))), # Get rid ofthis once data dict is no longer used
            options = [None] + list(np.arange(0,n_scans)), # Get rid ofthis once data dict is no longer used
            description = 'Select Fit Parameters',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '400px', margin = '0 0 5ps 0')
            )

        ### Build the Fitting Panel
        v1 = VBox([self.select_parameters_widget,self.BE_adjust_ref_widget,self.fit_method_widget])
        h1 = HBox([v1,self.spectra_to_fit_widget]) 
        
        save_h1 = HBox([self.save_params_name,self.save_params_button])
        save_h2 = HBox([self.save_fig_name,self.save_fig_button])
        v1_save = VBox([save_h1,save_h2])
        
        h2 = HBox([self.fit_button,self.plot_button,v1_save])
        
        vfinal = VBox([h1, h2, self.plotfit_chkbx, HBox([self.autofit_chkbx,self.autofit_button]), self.use_prev_fit_result_params, self.ref_lines_chkbx, self.plot_with_background_sub, \
                               self.plot_all_chkbx])
        
        display(vfinal)
        
        out = Output()
        display(out)


        # Button Actions
        @self.save_params_button.on_click
        def save_params_on_click(b):
            with out:
                if not hasattr(self,'fit_results'):
                    print('There are no fit result objects!')
                    return

                if not os.path.exists(os.path.join(os.getcwd(),'fit_parameters')):
                    os.makedirs(os.path.join(os.getcwd(),'fit_parameters'))

                # fit_params_list = [self.fit_results[i].params for i in self.fit_results_idx]
                fit_params_list = [j for j,x in enumerate(self.fit_object.fit_results) if x]
                # print(os.getcwd())
                save_location = os.path.join(os.getcwd(),'fit_parameters',self.save_params_name.value +'.pkl')
                f = open(save_location,"wb")

                pickle.dump(fit_params_list,f)
                f.close()

            
            
        @self.save_fig_button.on_click
        def save_fig_on_click(b):
            with out:
                if not hasattr(self,'fig'):
                    print('There is no figure object!')
                    return

                if not os.path.exists(os.path.join(os.getcwd(),'figures')):
                    os.makedirs(os.path.join(os.getcwd(),'figures'))
                    if not os.path.exists(os.path.join(os.getcwd(),'figures','fits')):
                        os.makedirs(os.path.join(os.getcwd(),'figures','fits'))

                save_location = os.path.join(os.getcwd(),'figures','fits',self.save_fig_name.value)          

                self.fig.savefig(save_location, bbox_inches='tight')



        
        @self.fit_button.on_click
        def fit_on_click(b):
            
            self.BE_adjust = 0  #change this once the new xps package is ready
            
            
            with out:
                # if hasattr(self,"fig"):
                clear_output(True)
                
                self.fit_spectra()
                show_inline_matplotlib_plots()
        
        
        
        @self.plot_button.on_click
        def plot_on_click(b):
            
            with out:
                # if hasattr(self,"fig"):
                clear_output(True)

                self.plot_spectra()
                show_inline_matplotlib_plots()    

                
        @self.autofit_button.on_click
        def plot_on_click(b):
            with out:
                if self.spectra_to_fit_widget.value == 'All':
                    specnum = 0
                else:
                    specnum = self.spectra_to_fit_widget.value
                print(specnum)
                if not hasattr(self,'autofit'):
                    self.autofit = XPyS.autofit.autofit.autofit(self.fit_object.esub,self.fit_object.isub[specnum[0]],self.fit_object.orbital)
                elif hasattr(self,'autofit'):
                    self.autofit.guess_params(energy = self.fit_object.esub,intensity = self.fit_object.isub[specnum[0]])
                for par in self.autofit.guess_pars.keys():
                    self.fit_object.params[par].value = self.autofit.guess_pars[par]




    # Fitting Panel Methods            
    def fit_spectra(self):
        ### Fitting Conditionals
        if self.fit_method_widget.value ==None:
            print('Enter a fitting Method',flush =True)
            return
        
        if (self.select_parameters_widget.value != None) & (not hasattr(self,'fit_results')):
            print('No Previous Fits!',flush =True)
            return
        
        
        if 'All' in self.spectra_to_fit_widget.value:
            print('%%%% Fitting all spectra... %%%%',flush =True)          
            fit_points = None
            
        elif self.spectra_to_fit_widget.value[0] == None:
#             self.fit_iter_idx = range(0)
            print('No Specta are selected!',flush =True)
            return
        else:
            fit_points = list(self.spectra_to_fit_widget.value)
            print('%%%% Fitting spectra ' + str(fit_points)+'... %%%%',flush =True) 

        self.fit_object.fit(specific_points = fit_points,plotflag = False, track = False, update_with_prev_pars = self.use_prev_fit_result_params.value,\
            autofit = self.autofit_chkbx.value)
        self.plot_spectra()


    def plot_spectra(self):
        ### Plotting Conditionals
        if self.plot_all_chkbx.value is True:
            print('Plotting all spectra ...')
            plot_points = [j for j,x in enumerate(self.fit_object.fit_results) if x]

        elif self.plot_all_chkbx.value is False:
            print('Plotting' + str(self.spectra_to_fit_widget.value) + ' spectra ...')

            if self.spectra_to_fit_widget.value[0] == None:
                print('Error!: You are trying to Not plot all results and Not fit any spectra')
                return

            elif 'All' in self.spectra_to_fit_widget.value:
                plot_points = [j for j,x in enumerate(self.fit_object.fit_results) if x]

            else:
                plot_points = dc(list(self.spectra_to_fit_widget.value))    
        print(plot_points)
        self.fit_object.plot_fitresults(specific_points = plot_points) 

    







class guipyter(spectra):

    def __init__(self,input_object):
        """Main class for performing interactive fitting of xps signal.


        Parameters
        ----------
        input_object: XPyS.sample.sample, list, dict
            The input object can be a few different things
            1. An sample object
            2. A list of sample objects. If this is the case then the sample_name attribute wil be used to select the samples
            3. A dictionary of sample objects where the key is the sample identifier. That sample identifier will be used to 
            select a given sample


        Returns
        -------


        See Also
        --------
        :func: plot_all_spectra(), plot_all_sub(), plot_atomic_percent()

        """
        input_object_list = []
        if type(input_object) is XPyS.sample.sample:
            self.samples = {}
            self.samples[input_object.sample_name] = input_object
            self.compound = False

        elif type(input_object) is list:
            self.samples = {s.sample_name:s for s in input_object}
            self.compound = False

        elif type(input_object) is dict:
            self.samples = input_object
            self.compound = False



        self.select_sample_widget = Dropdown(
            options = list(self.samples.keys()),
            description = 'Select Sample',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '200px', margin = '0 0 5ps 0')
            )

        self.select_compound_widget = Dropdown(
            options = [None] + [cmp for cmp in vars(self.samples[self.select_sample_widget.value]) if \
                type(self.samples[self.select_sample_widget.value].__dict__[cmp]) is XPyS.compound.CompoundSpectra],
            description = 'Select Compound',
            style = {'description_width': 'initial'},
            disabled=False,
            indent = False,
            layout = Layout(width = '200px', margin = '0 0 5ps 0')
            )
            
        self.select_spectra_widget = Dropdown(
            options = self.samples[self.select_sample_widget.value].element_scans,
            description = 'Select Spectra',
            style = {'description_width': 'initial'},
            disabled=False,
            indent = False,
            layout = Layout(width = '200px', margin = '0 0 5ps 0')
            )

        self.select_model_widget = Dropdown(
            options = [None] + XPyS.models.model_list(),
            description = 'Select Model',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '200px', margin = '0 0 5ps 0')
            )

        self.select_sample_model_widget = Dropdown(
            options = [None]+list(self.samples.keys()),
            description = 'Select Sample Model',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '200px', margin = '0 0 5ps 0')
            )

        self.spectra_to_compound_widget = SelectMultiple(
            options= self.samples[self.select_sample_widget.value].element_scans, 
            description='Compound',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '200px', margin = '0 0 5ps 0'),
            )

        self.select_spectra_button = Button(description="select spectra")
        
        self.compound_checkbox = Checkbox(
            value= False,
            description='Compound',
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            )  

        # self.select_spectra_to_compound_button = Button(description="Build Compound")

        # spectra_manager = VBox([self.select_spectra_widget,self.compound_checkbox])

        display(HBox([self.select_sample_widget, self.select_compound_widget,self.select_spectra_widget,self.select_model_widget,\
            self.select_sample_model_widget, self.select_spectra_button]))

        full_panel_out = Output()
        display(full_panel_out)

        @self.select_spectra_button.on_click
        def plot_on_click(b):

            with full_panel_out:
                clear_output(True)
                self.load_model = self.select_model_widget.value
                self.load_model_from_sample = self.select_sample_model_widget.value
                if (self.load_model != None) and (self.load_model_from_sample != None):
                    print('You cant choose to load a model from two places')
                else:
                    if self.select_compound_widget.value is None:
                        self.compound = False
                        self.create_full_panel(spectra_object = self.samples[self.select_sample_widget.value].__dict__[self.select_spectra_widget.value],compound = self.compound)
                    elif not self.select_compound_widget.value is None:
                        self.compound = True
                        self.compound_object = self.samples[self.select_sample_widget.value].__dict__[self.select_compound_widget.value]
                        self.create_full_panel(spectra_object = self.samples[self.select_sample_widget.value].__dict__[self.select_compound_widget.value].__dict__[self.select_spectra_widget.value],compound = self.compound)                

    def create_full_panel(self,spectra_object,compound):

        self.spectra_object = spectra_object
        
        if compound:
            other_spectra = [s_name for s_name in self.samples[self.select_sample_widget.value].__dict__[self.select_compound_widget.value].element_scans if not s_name in self.spectra_object.orbital]
            self.connected_spectra = {s_name:self.samples[self.select_sample_widget.value].__dict__[self.select_compound_widget.value].__dict__[s_name] for s_name in other_spectra}
            self.connected_prefixlist = {conn_spec[0]:[comp.prefix for comp in conn_spec[1].mod.components] for conn_spec in self.connected_spectra.items()}

        if self.load_model != None:
            ldd_mod = XPyS.models.load_model(self.load_model)
            self.spectra_object.mod = ldd_mod[0]
            self.spectra_object.params = ldd_mod[1]
            self.spectra_object.pairlist = ldd_mod[2]
            self.spectra_object.element_ctrl = ldd_mod[3]

        elif self.load_model_from_sample != None:
            self.spectra_object.mod = dc(self.samples[self.load_model_from_sample].__dict__[self.select_spectra_widget.value].mod)
            self.spectra_object.params = dc(self.samples[self.load_model_from_sample].__dict__[self.select_spectra_widget.value].params)
            self.spectra_object.pairlist = dc(self.samples[self.load_model_from_sample].__dict__[self.select_spectra_widget.value].pairlist)
            self.spectra_object.element_ctrl = dc(self.samples[self.load_model_from_sample].__dict__[self.select_spectra_widget.value].element_ctrl)


        self.E= spectra_object.E
        self.I= spectra_object.I
        self.esub= spectra_object.esub
        self.isub= spectra_object.isub
        self.prefixlist = [comp.prefix for comp in self.spectra_object.mod.components]

        
        # Make a list of all the relevant parameters. lmfit models can have parameters that depend on other parameters
        # and we are not interested in them. Next, make a list of the prefixes that we want a control bar for. This is a remnant 
        # of me using element_ctrl as a list of integers specifying which prefixes in pairlist* I want to control. 
        # element_ctrl could probably be replaced by a dict eventually. Last, create a dictionary of bool values for each parameter
        # to pass to ParameterWidgetGroup telling it to make a control slider or not. 

        # *pairlist is used to link different peaks together in an xps doublet, since you have to make different lmfit models for each peak
        

        self.rel_pars = [par for component_pars in [model_component._param_names for model_component in self.spectra_object.mod.components] \
            for par in component_pars]
        

    
        self.ctrl_prefixes = [[prefix for pairs in self.spectra_object.pairlist \
            for prefix in pairs][i] for i in self.spectra_object.element_ctrl]
        # print('Control Prefixes:', self.ctrl_prefixes)
        
        self.ctrl_pars = {par: any(x in par for x in self.ctrl_prefixes) for par in self.rel_pars}
        self.ctrl_lims = {}

        for par in self.rel_pars:
            if 'amplitude' in par:
                # self.ctrl_lims[par] = (0,15*np.max(self.spectra_object.params[par].value))
                self.ctrl_lims[par] = (0,5*np.max(self.isub))

            if 'center' in par:
                self.ctrl_lims[par] = ( np.min(self.E), np.max(self.E) )
            if 'sigma' in par:
                self.ctrl_lims[par] = ( 0, np.max([5,int(2*self.spectra_object.params[par].value)]) )
            if ('fraction' in par) or ('skew' in par):
                self.ctrl_lims[par] = (0,1)
        
        if self.compound:
            parameters = self.compound_object.params
            fitobject = self.compound_object
            n_scans = self.compound_object.n_scans
        elif not self.compound:
            parameters = self.spectra_object.params
            fitobject = self.spectra_object
            n_scans = len(self.spectra_object.isub)

        self.make_parameter_panel(parameters = parameters)
        self.make_interactive_plot()
        self.fitting_panel = fitting_panel(fitobject,n_scans)
    

    def make_parameter_panel(self,parameters = None):
        if parameters == None:
            print('No parameters specified')
            return
        box_layout = Layout(display='flex',
                                    flex_flow='column',
                                    align_items='stretch',
                                    width='100%')
        self.paramwidgetscontainer = VBox([], layout=box_layout)

        if self.spectra_object.pairlist is not None:   #### This is here so that pairlists dont need to be specified for models in future development

                
            self.paramwidgets = {p_name:ParameterWidgetGroup(p,slider_ctrl = self.ctrl_pars[p_name],sliderlims = self.ctrl_lims[p_name])\
                 for p_name, p in parameters.items() if p_name in self.rel_pars}

            # for pw in self.paramwidgets.values():
            #     pw.value_text.observe(lambda e: self.update_plot(), names='value')

            ### The children are the paramwidgets for each model
            self.paramwidgetscontainer.children = [HBox([self.paramwidgets[comp_name].get_widget() \
                for comp_name in self.spectra_object.mod.components[i]._param_names]) for i in range(len(self.spectra_object.mod.components))]
            display(self.paramwidgetscontainer)


    # def update_parameter_panel(self,parameters = None):
    #     self.paramwidgets = {p_name:ParameterWidgetGroup(p,slider_ctrl = self.ctrl_pars[p_name],sliderlims = self.ctrl_lims[p_name])\
    #              for p_name, p in parameters.items() if p_name in self.rel_pars}


    def make_interactive_plot(self):

        self.change_pars_to_fit_button = Button(
            description="Change Parameters to Fit Result",
            layout = Layout(width = '300px', margin = '0 0 5ps 0')
            ) 

        self.reset_slider_lims_button = Button(
            description="Reset Slider Max",
            layout = Layout(width = '300px', margin = '0 0 5ps 0')
            ) 

        self.data_init_widget =  Dropdown(
            options=list(np.arange(0,len(self.isub))),      
            value = 0,
            description='Data to initialize',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '200px', margin = '0 0 5ps 0')
            )

        self.reset_slider_widget =  Dropdown(
            options= [par for par in self.ctrl_pars.keys() if self.ctrl_pars[par]],  
            # options=list(np.arange(0,len(self.isub))),     
            value = None,
            description='Slider Reset',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '200px', margin = '0 0 5ps 0')
            )        

        self.wlim = {}
        self.wlim[self.spectra_object.orbital] = FloatRangeSlider (
                value=[np.min(self.esub), np.max(self.esub)],
                min = np.min(self.esub),
                max =np.max(self.esub),
                step  = 0.01,
                description = self.spectra_object.orbital+'_xlim',
                style = {'description_width': 'initial'},
                layout = Layout(width = '300px', margin = '0 0 5ps 0')
                )      
        if self.compound:
            for orbital in self.connected_spectra.keys():
                self.wlim[orbital] = FloatRangeSlider (
                        value=[np.min(self.connected_spectra[orbital].esub), np.max(self.connected_spectra[orbital].esub)],
                        min = np.min(self.connected_spectra[orbital].esub),
                        max =np.max(self.connected_spectra[orbital].esub),
                        step  = 0.01,
                        description = orbital+'_xlim',
                        style = {'description_width': 'initial'},
                        layout = Layout(width = '300px', margin = '0 0 5ps 0')
                        )    

        out = Output()
        display(out)

        @self.change_pars_to_fit_button.on_click
        def plot_on_click(b):
            with out:
                if not self.compound:
                    self.spectra_object.params = self.spectra_object.fit_results[self.data_init_widget.value].params.copy()
                    for pars in self.paramwidgets.keys():
                        self.paramwidgets[pars].update_widget_group(self.spectra_object.params[pars])
                elif self.compound:
                    self.compound_object.params = self.compound_object.fit_results[self.data_init_widget.value].params.copy()
                    for pars in self.paramwidgets.keys():
                        self.paramwidgets[pars].update_widget_group(self.compound_object.params[pars])     

        @self.reset_slider_lims_button.on_click
        def plot_on_click(b):
            with out:
                if self.reset_slider_widget.value !=None:
                    self.paramwidgets[self.reset_slider_widget.value].ctrl_slider.max  = 2*self.paramwidgets[self.reset_slider_widget.value].ctrl_slider.value

        # Create the interactive plot, then build the slider/graph parameter controls
        plotkwargs = {**{pw.name:pw.ctrl_slider for pw in self.paramwidgets.values() if hasattr(pw,'ctrl_slider')},\
            **{plotlim.description:plotlim for plotlim in self.wlim.values()}}
        self.intplot = interactive(self.interactive_plot,**plotkwargs)
        
        vb = VBox(self.intplot.children[0:-1])
        vb2 = VBox([HBox([VBox([self.data_init_widget,self.reset_slider_widget]),VBox([self.change_pars_to_fit_button,self.reset_slider_lims_button])]),self.intplot.children[-1]])
        hb = HBox([vb,vb2])
            
        display(hb)

    def combine_components(self,pair,spectra_object,prefixlist,energy):
        """Functinon for combining doublets"""
        tempindex = np.empty(len(pair))
        
        for i in range(len(pair)):
            tempindex[i] = prefixlist.index(pair[i])
        
        if not self.compound:
            combined_comps = [spectra_object.mod.components[int(tempindex[i])].eval(spectra_object.params,x=energy) for i in range(len(tempindex))]
        elif self.compound:
            combined_comps = [spectra_object.mod.components[int(tempindex[i])].eval(self.compound_object.params,x=energy) for i in range(len(tempindex))]
        return [sum(x) for x in zip(*combined_comps)]
        
        
        

    def interactive_plot(self,*args,**kwargs):
        """interactive plotting function to be called by ipywidget.interactive"""        


        if not self.compound:
            fig,ax = plt.subplots(figsize=(8,6))
            p1 = ax.plot(self.esub,self.isub[self.data_init_widget.value],'bo')
            p2 = ax.plot(self.esub,self.spectra_object.mod.eval(self.spectra_object.params,x=self.esub) , color = 'black')
        
            p = [[] for i in range(len(self.spectra_object.pairlist))]
            fit_legend = [element_text[element[0]] for element in self.spectra_object.pairlist]
            
            for pairs in enumerate(self.spectra_object.pairlist):

                p[pairs[0]] = ax.fill_between(self.esub,self.combine_components(pairs[1],self.spectra_object,self.prefixlist,energy = self.esub),\
                                            color = element_color[pairs[1][0]],alpha = 0.3)

                # p[pairs[0]] = plt.fill_between(self.E,sum([self.spectra_object.mod.eval_components(x=self.E)[comp] for comp in pairs[1]]),\
                #                                color = element_color[pairs[1][0]],alpha = 0.3)
                                            
                                                    
                
            ax.set_xlim(np.max(kwargs[self.spectra_object.orbital+'_xlim']),np.min(kwargs[self.spectra_object.orbital+'_xlim']))

            ax.legend(p,fit_legend,bbox_to_anchor=(0.5, 1.05), loc='lower center')
            
            plt.show()

        elif self.compound:
            n_plots = len(self.connected_spectra.keys())+1
            fig,ax = plt.subplots(n_plots,1,figsize=(8,n_plots*4))
            p1 = ax[0].plot(self.esub,self.isub[self.data_init_widget.value],'bo')
            p2 = ax[0].plot(self.esub,self.spectra_object.mod.eval(self.compound_object.params,x=self.esub) , color = 'black')


            p = [[] for i in range(len(self.spectra_object.pairlist))]
            fit_legend = [element_text[element[0]] for element in self.spectra_object.pairlist]
            for pairs in enumerate(self.spectra_object.pairlist):
                p[pairs[0]] = ax[0].fill_between(self.esub,self.combine_components(pairs[1],self.spectra_object,self.prefixlist,energy = self.esub),\
                                            color = element_color[pairs[1][0]],alpha = 0.3)

            ax[0].set_xlim(np.max(kwargs[self.spectra_object.orbital+'_xlim']),np.min(kwargs[self.spectra_object.orbital+'_xlim']))


            for orb in enumerate(self.connected_spectra.keys()):
                energy = self.connected_spectra[orb[1]].esub
                spec_obj = self.connected_spectra[orb[1]]
                ax[orb[0]+1].plot(energy,spec_obj.isub[self.data_init_widget.value],'bo')
                ax[orb[0]+1].plot(energy,spec_obj.mod.eval(self.compound_object.params,x=energy) , color = 'black')

            
                for pairs in enumerate(spec_obj.pairlist):

                    ax[orb[0]+1].fill_between(energy,self.combine_components(pairs[1],spec_obj,self.connected_prefixlist[orb[1]],energy = energy),\
                                                color = element_color[pairs[1][0]],alpha = 0.3)
                                  
                                                    
                ax[orb[0]+1].set_xlim(np.max(kwargs[spec_obj.orbital+'_xlim']),np.min(kwargs[spec_obj.orbital+'_xlim']))


            ax[0].legend(p,fit_legend,bbox_to_anchor=(0.5, 1.05), loc='lower center')
            
            plt.show()

