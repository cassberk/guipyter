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
# sys.path.append("/Volumes/GoogleDrive/My Drive/XPS/XPS_Library")

import xps_peakfit
from xps_peakfit import bkgrds as background_sub
from xps_peakfit.helper_functions import *
from xps_peakfit.gui_element_dicts import *

from xps_peakfit.spectra import spectra
# import xps_peakfit.auto_fitting
import os
import glob



class ParameterWidgetGroup:
    """Modified from existing lmfit.ui.ipy_fitter"""
    def __init__(self, par, slider_ctrl=True,sliderlims = None):
        self.par = par
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
        if slider_ctrl is True:

            self.ctrl_slider = FloatSlider (
                        value=self.par.value,
                        min = sliderlims[0],
                        max = sliderlims[1], ### Need to figure out a way to set this
                        step  = 0.01,
                        description = self.par.name,
                        style = {'description_width': 'initial','handle_color' : 'blue'},
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
        self.min_text.observe(self._on_min_value_change, names='value')
        self.max_text.observe(self._on_max_value_change, names='value')
        # self.min_checkbox.observe(self._on_min_checkbox_change, names='value')
        # self.max_checkbox.observe(self._on_max_checkbox_change, names='value')
        self.vary_checkbox.observe(self._on_vary_change, names='value')

    def _on_value_change(self, change):
        self.par.set(value=change['new'])

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

    # # Make it easy to set the widget attributes directly.
    # @property
    # def value(self):
    #     return self.value_text.value

    # @value.setter
    # def value(self, value):
    #     self.value_text.value = value

    # @property
    # def vary(self):
    #     return self.vary_checkbox.value

    # @vary.setter
    # def vary(self, value):
    #     self.vary_checkbox.value = value

    # @property
    # def min(self):
    #     return self.min_text.value

    # @min.setter
    # def min(self, value):
    #     self.min_text.value = value

    # @property
    # def max(self):
    #     return self.max_text.value

    # @max.setter
    # def max(self, value):
    #     self.max_text.value = value

    @property
    def name(self):
        return self.par.name


class fitting_panel:

    def __init__(self, spectra_object):
        self.spectra_object = spectra_object

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
        self.plot_loaded_parameters_button = Button(description="Plot Loaded Parameters") 
          
        
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
            options=[None, 'All'] + list(np.arange(0,len(self.spectra_object.isub))), # Get rid ofthis once data dict is no longer used
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

        self.use_input_params = Checkbox(
            value= False,
            description='Initialize with param list',
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
            options = [None] + list(np.arange(0,len(self.spectra_object.isub))), # Get rid ofthis once data dict is no longer used
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
        
        h2 = HBox([self.fit_button,self.plot_button,self.plot_loaded_parameters_button,v1_save])
        
        vfinal = VBox([h1, h2, self.plotfit_chkbx, self.autofit_chkbx, self.use_input_params, self.ref_lines_chkbx, self.plot_with_background_sub, \
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
                fit_params_list = [j for j,x in enumerate(self.spectra_objet.fit_results) if x]
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

                self.plot()
                show_inline_matplotlib_plots()    

                
        @self.plot_loaded_parameters_button.on_click
        def plot_on_click(b):
            
            with out:
                # if hasattr(self,"fig"):
                clear_output(True)

                self.plot_input_parameters()
                show_inline_matplotlib_plots()  
                

    # Fitting Panel Methods            
    def fit_spectra(self):
        ### Fitting Conditionals
        if self.fit_method_widget.value ==None:
            print('Enter a fitting Method')
            return
        
        if (self.select_parameters_widget.value != None) & (not hasattr(self,'fit_results')):
            print('No Previous Fits!')
            return
        
        
        if 'All' in self.spectra_to_fit_widget.value:
            print('%%%% Fitting all spectra... %%%%')            
            fit_points = None
            
        elif self.spectra_to_fit_widget.value[0] == None:
#             self.fit_iter_idx = range(0)
            print('No Specta are selected!')
            return
        else:
            fit_points = list(self.spectra_to_fit_widget.value)
            print('%%%% Fitting spectra ' + str(fit_points)+'... %%%%') 

        self.spectra_object.fit(specific_points = fit_points,plotflag = False, track = False)
        self.plot_spectra()


    def plot_spectra(self):
        ### Plotting Conditionals
        if self.plot_all_chkbx.value is True:
            print('Plotting all spectra ...')
            plot_points = [j for j,x in enumerate(self.spectra_object.fit_results) if x]

        elif self.plot_all_chkbx.value is False:
            print('Plotting' + str(self.spectra_to_fit_widget.value) + ' spectra ...')

            if self.spectra_to_fit_widget.value[0] == None:
                print('Error!: You are trying to Not plot all results and Not fit any spectra')
                return

            elif 'All' in self.spectra_to_fit_widget.value:
                plot_points = [j for j,x in enumerate(self.fit_results) if x]

            else:
                plot_points = dc(list(self.spectra_to_fit_widget.value))    
        
        self.spectra_object.plot_fitresults(specific_points = plot_points) 

    







class guipyter(spectra):

    def __init__(self,spectra_object,orbital=None,parameters=None,model=None,pairlist=None,element_ctrl=None,\
        spectra_name = None, carbon_adjust=None,load_spectra_object = False,load_model = False,autofit = False):

        # xps_spec.__init__(self,sample_object,orbital,parameters,model,pairlist,element_ctrl,\
        # spectra_name, carbon_adjust,load_spectra_object,load_model,autofit)
        self.spectra_object = spectra_object
        if load_model:
            f = open('/Volumes/GoogleDrive/My Drive/XPS/XPS_Library/xps/models/load_model_info.pkl', 'rb')   # 'r' for reading; can be omitted
            load_dict = pickle.load(f)         # load file content as mydict
            f.close() 

            self.spectra_object.mod = lm.model.load_model(load_dict[load_model]['model_path'])
            self.spectra_object.params = pickle.load(open(load_dict[load_model]['params_path'],"rb"))[0]
            self.spectra_object.pairlist = load_dict[load_model]['pairlist']
            self.spectra_object.element_ctrl = load_dict[load_model]['element_ctrl']
        else:
            self.spectra_object.mod = spectra_object.mod
            self.spectra_object.element_ctrl = spectra_object.element_ctrl
            self.spectra_object.pairlist = spectra_object.pairlist
            self.spectra_object.params = spectra_object.params

        self.E= spectra_object.E
        self.I= spectra_object.I
        self.esub= spectra_object.esub
        self.isub= spectra_object.isub
        self.prefixlist = [comp.prefix for comp in self.spectra_object.mod.components]
        # spectra_object.mod = self.spectra_object.mod
        # spectra_object.params = self.spectra_object.params
        # spectra_object.pairlist = self.spectra_object.pairlist
        # spectra_object.element_ctrl = self.spectra_object.element_ctrl
        # self.prefixlist= spectra_object.prefixlist
        
        # Make a list of all the relevant parameters. lmfit models can have parameters that depend on other parameters
        # and we are not interested in them. Next, make a list of the prefixes that we want a control bar for. This is a remnant 
        # of me using element_ctrl as a list of integers specifying which prefixes in pairlist* I want to control. 
        # element_ctrl could probably be replaced by a dict eventually. Last, create a dictionary of bool values for each parameter
        # to pass to ParameterWidgetGroup telling it to make a control slider or not. 

        # *pairlist is used to link different peaks together in an xps doublet, since you have to make different lmfit models for each peak
        

        self.rel_pars = [par for component_pars in [model_component._param_names for model_component in self.spectra_object.mod.components] \
            for par in component_pars]
    
        ctrl_prefixes = [[prefix for pairs in self.spectra_object.pairlist \
            for prefix in pairs][i] for i in self.spectra_object.element_ctrl]
        print(ctrl_prefixes)
        
        self.ctrl_pars = {par: any(x in par for x in ctrl_prefixes) for par in self.rel_pars}
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
        
        self.parameter_panel(parameters = self.spectra_object.params)
        self.make_interactive_plot()
        self.fitting_panel = fitting_panel(self.spectra_object)


    def parameter_panel(self,parameters = None):
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

            """The children are the paramwidgets for each model"""
            self.paramwidgetscontainer.children = [HBox([self.paramwidgets[comp_name].get_widget() \
                for comp_name in self.spectra_object.mod.components[i]._param_names]) for i in range(len(self.spectra_object.mod.components))]
            display(self.paramwidgetscontainer)


    def make_interactive_plot(self):

        self.change_pars_to_fit_button = Button(
            description="Change Parameters to Fit Result",
            layout = Layout(width = '300px', margin = '0 0 5ps 0')
            ) 

        self.data_init_widget =  Dropdown(
            # options=list(np.arange(0,len(self.data['isub']))),      # Get rid ofthis once data dict is no longer used
            options=list(np.arange(0,len(self.isub))),      
            value = 0,
            description='Data to initialize',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '200px', margin = '0 0 5ps 0')
            )
        
        ### Created Now, but used in the interactive fitting function
        # self.select_parameters_widget = Dropdown(
        #     options = [None] + list(np.arange(0,len(self.data['isub']))),
        #     description = 'Select Fit Parameters',
        #     style = {'description_width': 'initial'},
        #     disabled=False,
        #     layout = Layout(width = '400px', margin = '0 0 5ps 0')
        #     )

        
        wlim = FloatRangeSlider (
                value=[np.min(self.esub), np.max(self.esub)],
                min = np.min(self.esub),
                max =np.max(self.esub),
                step  = 0.01,
                description = 'xlim',
                style = {'description_width': 'initial'},
                layout = Layout(width = '300px', margin = '0 0 5ps 0')
                )      
    
        out = Output()
        display(out)

        @self.change_pars_to_fit_button.on_click
        def plot_on_click(b):
            with out:

                for pars in self.paramwidgets.keys():

                    if self.spectra_object.fit_results[self.data_init_widget.value].params[pars].expr == None:
                        # self.paramwidgets[pars].expr_text.value = dc(self.spectra_object.fit_results[self.data_init_widget.value].params[pars].expr)
                        self.paramwidgets[pars].value_text.value = dc(self.spectra_object.fit_results[self.data_init_widget.value].params[pars].value)


        # Create the interactive plot, then build the slider/graph parameter controls
        plotkwargs = {**{pw.name:pw.ctrl_slider for pw in self.paramwidgets.values() if hasattr(pw,'ctrl_slider')},\
            **{wlim.description:wlim}}
        self.intplot = interactive(self.interactive_plot,**plotkwargs)
        
        vb = VBox(self.intplot.children[0:-1])
        vb2 = VBox([HBox([self.data_init_widget,self.change_pars_to_fit_button]),self.intplot.children[-1]])
        hb = HBox([vb,vb2])
            
        display(hb)

    def combine_components(self,pair):
        """Functinon for combining doublets"""
        tempindex = np.empty(len(pair))
        
        for i in range(len(pair)):
            tempindex[i] = self.prefixlist.index(pair[i])
                    
        combined_comps = [self.spectra_object.mod.components[int(tempindex[i])].eval(self.spectra_object.params,x=self.esub) for i in range(len(tempindex))]
        return [sum(x) for x in zip(*combined_comps)]
        
        
        

    def interactive_plot(self,*args,**kwargs):
        """interactive plotting function to be called by ipywidget.interactive"""        

        plt.figure(figsize=(8,6))
        p1 = plt.plot(self.esub,self.isub[self.data_init_widget.value],'bo')
        p2 = plt.plot(self.esub,self.spectra_object.mod.eval(self.spectra_object.params,x=self.esub) , color = 'black')

        
        
        p = [[] for i in range(len(self.spectra_object.pairlist))]
        fit_legend = [element_text[element[0]] for element in self.spectra_object.pairlist]
        
        for pairs in enumerate(self.spectra_object.pairlist):

            p[pairs[0]] = plt.fill_between(self.esub,self.combine_components(pairs[1]),\
                                           color = element_color[pairs[1][0]],alpha = 0.3)
            # p[pairs[0]] = plt.fill_between(self.E,sum([self.spectra_object.mod.eval_components(x=self.E)[comp] for comp in pairs[1]]),\
            #                                color = element_color[pairs[1][0]],alpha = 0.3)
                                           
                                                  
            
        plt.xlim(np.max(kwargs['xlim']),np.min(kwargs['xlim']))

        plt.legend(p,fit_legend,bbox_to_anchor=(0.5, 1.05), loc='lower center')

        plt.show()



### This is in order to use this while using the matplotlib widget backend not developed yet
#     def plot_widget(self):
#         self.fig, self.ax = plt.subplots()
#         self.ax.plot(self.E,self.I[self.data_init_widget.value] label='data')
#         self.ax.plot(self.E,self.spectra_object.mod.eval(self.spectra_object.params,x=self.E) , color = 'black', label='model')
#         self.ax.legend()
#         return self.fig.canvas

#     def update_plot(self):
#         self.ax.clear()
#         self.ax.plot(self.x, self.data, label='data')

#         numsteps = 1000
#         xmod = np.linspace(self.x.min(), self.x.max(), numsteps)
#         if self.spectra_object.mod is None:
#             self.ax.plot(xmod, np.zeros_like(xmod), label='model')
#         else:
#             try:
#                 self.ax.plot(xmod, self.fit.eval(params=self.pars, x=xmod), label='model')
#                 for c, v in self.fit.eval_components(params=self.pars, x=xmod).items():
#                     plt.plot(xmod, v, '--', label=c)
#             except AttributeError:
#                 self.ax.plot(xmod, self.spectra_object.mod.eval(params=self.pars, x=xmod), label='model')
#                 for c, v in self.spectra_object.mod.eval_components(params=self.pars, x=xmod).items():
#                     plt.plot(xmod, v, '--', label=c)

#         self.ax.legend()
#         self.fig.canvas.draw()
# #         self.fig.canvas.flush_events()