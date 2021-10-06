# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:36:04 2021

@author: deanp

Evaluating Passes

This code closely follows that of David Sumpter's xG model code in the following
repo: 
"""

#%%
import pandas as pd
import numpy as np
import json
import FCPython
import matplotlib.pyplot as plt
#%%
#Load Data (Premier League 2017-2018)
with open('C:/Users/deanp/OneDrive/Desktop/Football Analytics/Modelling Football Course/Data/Wyscout/events_England.json') as f:
    data_england = json.load(f)

#Load Data (La Liga 2017-2018)
with open('C:/Users/deanp/OneDrive/Desktop/Football Analytics/Modelling Football Course/Data/Wyscout/events_Spain.json') as f:
    data_spain = json.load(f)

#Load Data (World Cup 2018)
with open('C:/Users/deanp/OneDrive/Desktop/Football Analytics/Modelling Football Course/Data/Wyscout/events_World_Cup.json') as f:
    data_wc = json.load(f)
#%%
#Create a data set of passes
df_england = pd.DataFrame(data_england)
df_spain = pd.DataFrame(data_spain)
df_wc = pd.DataFrame(data_wc)

#%%
def clean_data(df):
    pd.unique(df['subEventName'])
    passes=df[df['eventName'] == 'Pass']
    #Filter out Head passes, Launches
    passes=passes[passes['subEventName'] != 'Launch']
    return passes

#%%
england_passes = clean_data(df_england)
spain_passes = clean_data(df_spain)
wc_passes = clean_data(df_wc)

#%%
#Set up models
pass_model_england=pd.DataFrame(columns=['Success','X start','Y start'])
pass_model_spain=pd.DataFrame(columns=['Success','X start','Y start'])
pass_model_wc=pd.DataFrame(columns=['Success','X start','Y start'])

#%%

def add_features(pass_model, passes):
    passes_dict = passes.to_dict('records')
    i = 0
    for pass_made in passes_dict:
        pass_model.at[i,'X start']=100-pass_made['positions'][0]['x']
        pass_model.at[i,'Y start']=pass_made['positions'][0]['y']
        pass_model.at[i,'C']=abs(pass_made['positions'][0]['y']-50)
        
        x=pass_model.at[i,'X start']*105/100
        y=pass_model.at[i,'C']*65/100
        pass_model.at[i,'Distance']=np.sqrt(x**2 + y**2)
        #Was it successful?
        pass_model.at[i,'Success']=0
        if pass_made['tags'][0]['id']==1801:
            pass_model.at[i,'Success']=1
        i += 1
    return pass_model

#%%
pass_model_england = add_features(pass_model_england, england_passes)
#%%
pass_model_spain = add_features(pass_model_spain, spain_passes)
#%%
pass_model_wc = add_features(pass_model_wc, wc_passes)

#%%
#Visualizing proportion of successful to unsuccessful passes
def donut_successful_vs_unsuccessful(pass_model):
    num_success = len(pass_model[pass_model['Success'] == 1])
    num_not_success = len(pass_model) - num_success
    # create data
    names = ['Successful', 'Unsuccessful']
    sizes = np.array([num_success, num_not_success])
     
    # Create a circle at the center of the plot
    my_circle = plt.Circle( (0,0), 0.7, color='white')
    
    # Give color names
    plt.pie(sizes, labels=names, colors=['#567899', '#98A9BA'])
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    
    # Show the graph
    plt.show()
    return plt

#%%
donut_successful_vs_unsuccessful(pass_model_england)
donut_successful_vs_unsuccessful(pass_model_spain)
donut_successful_vs_unsuccessful(pass_model_wc)

#%%
#produce 2D histogram for heat map
def create_histogram(pass_model):
    H_pass=np.histogram2d(pass_model['X start'], pass_model['Y start'],bins=50,range=[[0, 100],[0, 100]])
    successful_passes_only=pass_model[pass_model['Success']==1]
    H_successful_pass=np.histogram2d(successful_passes_only['X start'], successful_passes_only['Y start'],bins=50,range=[[0, 100],[0, 100]])
    return H_pass, H_successful_pass

#%%
#2D histograms
H_pass_ENG, H_success_ENG = create_histogram(pass_model_england)
H_pass_ESP, H_success_ESP = create_histogram(pass_model_spain)
H_pass_WC, H_success_WC = create_histogram(pass_model_wc)

#%%
def plot_heatmap_passes(H_pass, country, attempted_or_successful):
    (fig,ax) = FCPython.createGoalMouth()
    pos=ax.imshow(H_pass[0], extent=[-1,66,104,-1], aspect='auto',cmap=plt.cm.Reds)
    fig.colorbar(pos, ax=ax)
    ax.set_title(country + ': Number of Passes ' + attempted_or_successful)
    plt.xlim((-1,66))
    plt.ylim((-3,35))
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    fig.savefig('C:/Users/deanp/OneDrive/Desktop/Football Analytics/Output/NumPasses' + country + '.png', dpi=200, bbox_inches='tight') 
    return fig, ax

#%%
#plot passes attempted, successful passes
plot_heatmap_passes(H_pass_ENG, "ENG", "Attempted")
plot_heatmap_passes(H_success_ENG, "ENG", "Successful")
plot_heatmap_passes(H_pass_ESP, "ESP", "Attempted")
plot_heatmap_passes(H_success_ESP, "ESP", "Successful")
plot_heatmap_passes(H_pass_WC, "WC2018", "Attempted")
plot_heatmap_passes(H_success_WC, "WC2018", "Successful")

#%%
#Plot a logistic curve
# b=[3, -3]
# x=np.arange(5,step=0.1)
# y=1/(1+np.exp(-b[0]-b[1]*x))
# fig,ax=plt.subplots(num=1)
# plt.ylim((-0.05,1.05))
# plt.xlim((0,5))
# ax.set_ylabel('y')
# ax.set_xlabel("x") 
# ax.plot(x, y, linestyle='solid', color='black')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.show()


#%%
#Function to Plot some column X of passes against successfulness
def plot_pass_success(pass_model, column_header, country):
    fig,ax=plt.subplots(num=1)
    ax.plot(pass_model[column_header], pass_model['Success'], linestyle='none', marker= '.', markerSize= 12, color='black')
    ax.set_ylabel(country + ': Successful pass')
    ax.set_xlabel("Pass " + column_header)
    plt.ylim((-0.05,1.05))
    ax.set_yticks([0,1])
    ax.set_yticklabels(['No','Yes'])
    plt.show()
    return fig, ax

#%%
# plot distance vs success
plot_pass_success(pass_model_england, 'Distance', "ENG")
plot_pass_success(pass_model_spain, 'Distance', "ESP")
plot_pass_success(pass_model_wc, 'Distance', "WC2018")

#%%

def plot_distance_against_probabilitySuccess(pass_model, country, bins):
    successful_passes_only=pass_model[pass_model['Success']==1]
    passcount_dist=np.histogram(pass_model['Distance'],bins=bins,range=[0, 100])
    successful_passcount_dist=np.histogram(successful_passes_only['Distance'],bins=bins,range=[0, 100])
    prob_succesful_pass=np.divide(successful_passcount_dist[0],passcount_dist[0])
    distance=passcount_dist[1]
    middistance= (distance[:-1] + distance[1:])/2
    fig,ax=plt.subplots(num=1)
    ax.plot(middistance, prob_succesful_pass, linestyle='none', marker= '.', color='black')
    ax.set_ylabel(country + ': Probability pass completed')
    ax.set_xlabel("Distance from goal (metres)")
    ax.set_yticks([0,0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    return fig, ax
#%%
plot_distance_against_probabilitySuccess(pass_model_england, "ENG", 60)
plot_distance_against_probabilitySuccess(pass_model_spain, "ESP", 60)
plot_distance_against_probabilitySuccess(pass_model_wc, "WC2018", 60)

#%%
#REGRESSION MODELS
import statsmodels.api as sm
import statsmodels.formula.api as smf

#%%
model_variables = ['Distance']

def create_model(model_variables, pass_model):
    features=''
    for v in model_variables[:-1]:
        features = features  + v + ' + '
    features = features + model_variables[-1]
    model = smf.glm(formula="Success ~ " + features, data=pass_model, 
                           family=sm.families.Binomial()).fit()
    return model

def get_model_params(model):
    return model.params

eng_model = create_model(model_variables, pass_model_england)
b=get_model_params(eng_model)

#%%
#Return xSuccessPass value for more general model
def calculate_xSuccessfulPass(pass_):
   bsum=b[0] #intercept
   for i,v in enumerate(model_variables):
       bsum=bsum+b[i+1]*pass_[v]
   xSuccessPass = 1/(1+np.exp(bsum)) 
   return xSuccessPass

#%%
#Add an xSuccessPass column to original dataframe
def add_xSuccessPass(model, pass_model):
    xSuccessPass=pass_model.apply(calculate_xSuccessfulPass, axis=1) 
    pass_model = pass_model.assign(xSuccessPass=xSuccessPass)
    return pass_model

#%%

