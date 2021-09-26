# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:36:04 2021

@author: deanp

Evaluating Passes
"""

#%%
import pandas as pd
import numpy as np
import json
import FCPython



#%%
#Load Data (Premier League 2017-2018)
with open('C:/Users/deanp/OneDrive/Desktop/Football Analytics/Modelling Football Course/Data/Wyscout/events_England.json') as f:
    data = json.load(f)

#%%
#Create a data set of passes
train = pd.DataFrame(data)
pd.unique(train['subEventName'])
passes=train[train['eventName'] == 'Pass']

#%%
#Filter out Head passes, Launches
passes=passes[passes['subEventName'] != 'Head pass']
passes=passes[passes['subEventName'] != 'Launch']

#%%
#Set up model
pass_model=pd.DataFrame(columns=['Success','X start','Y start'])

#%%
#Go through the dataframe and calculate the X, Y start co-ordinates.
for i,pass_made in passes.iterrows():
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

# passes in the 

#%%
#Visualizing proportion of successfu to unsuccessful passes
num_success = len(pass_model[pass_model['Success'] == 1])
num_not_success = len(pass_model) - num_success

import matplotlib.pyplot as plt

y = np.array([num_success, num_not_success])

plt.pie(y)
plt.show() 

#%%
#Get first 500 passes
passes_500=pass_model.iloc[:500]

#%%
#2D histogram
H_pass=np.histogram2d(pass_model['X start'], pass_model['Y start'],bins=50,range=[[0, 100],[0, 100]])
successful_passes_only=pass_model[pass_model['Success']==1]
H_successful_pass=np.histogram2d(successful_passes_only['X start'], successful_passes_only['Y start'],bins=50,range=[[0, 100],[0, 100]])

#%%
#Plot number of passes
(fig,ax) = FCPython.createGoalMouth()
pos=ax.imshow(H_pass[0], extent=[-1,66,104,-1], aspect='auto',cmap=plt.cm.Reds)
fig.colorbar(pos, ax=ax)
ax.set_title('Number of passes (opposition goal mouth)')
plt.xlim((-1,66))
plt.ylim((-3,35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#%%
#Plot number of SUCCESFUL passes
(fig,ax) = FCPython.createGoalMouth()
pos=ax.imshow(H_successful_pass[0], extent=[-1,66,104,-1], aspect='auto',cmap=plt.cm.Reds)
fig.colorbar(pos, ax=ax)
ax.set_title('Number of succesful passes (opposition goal mouth)')
plt.xlim((-1,66))
plt.ylim((-3,35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#%%
#Plot a logistic curve
b=[3, -3]
x=np.arange(5,step=0.1)
y=1/(1+np.exp(-b[0]-b[1]*x))
fig,ax=plt.subplots(num=1)
plt.ylim((-0.05,1.05))
plt.xlim((0,5))
ax.set_ylabel('y')
ax.set_xlabel("x") 
ax.plot(x, y, linestyle='solid', color='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


#%%
#Function to Plot some column of first 500 passes against successfulness
def plot_pass_success(column_header):
    fig,ax=plt.subplots(num=1)
    ax.plot(passes_500[column_header], passes_500['Success'], linestyle='none', marker= '.', markerSize= 12, color='black')
    ax.set_ylabel('Successful pass')
    ax.set_xlabel("Pass " + column_header)
    plt.ylim((-0.05,1.05))
    ax.set_yticks([0,1])
    ax.set_yticklabels(['No','Yes'])
    plt.show()

#%%
# plot distance vs success
plot_pass_success('Distance')
# FINDING: ^ no successful passes within 20 meters of the goal?

#%%
#Plot distance vs probability of success 

passcount_dist=np.histogram(pass_model['Distance'],bins=40,range=[0, 100])
successful_passcount_dist=np.histogram(successful_passes_only['Distance'],bins=40,range=[0, 100])
prob_succesful_pass=np.divide(successful_passcount_dist[0],passcount_dist[0])
distance=passcount_dist[1]
middistance= (distance[:-1] + distance[1:])/2
fig,ax=plt.subplots(num=1)
ax.plot(middistance, prob_succesful_pass, linestyle='none', marker= '.', color='black')
ax.set_ylabel('Probability pass completed')
ax.set_xlabel("Distance from goal (metres)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# FINDING: probability of pass success seems to drop off once in final third

#%%
#Statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf
#Make single variable model of distance
distance_only_model = smf.glm(formula="Success ~ Distance" , data=pass_model, 
                           family=sm.families.Binomial()).fit()
print(distance_only_model.summary())  
b=distance_only_model.params
pass_success_prob=1/(1+np.exp(b[0]+b[1]*middistance)) 
ax.plot(middistance, pass_success_prob, linestyle='solid', color='black')
plt.show()

#%%
model_variables = ['Distance']

#Return xG value for more general model
def calculate_xSuccessfulPass(pass_):    
   bsum=b[0] #intercept
   for i,v in enumerate(model_variables):
       bsum=bsum+b[i+1]*pass_[v]
   xSuccessPass = 1/(1+np.exp(bsum)) 
   return xSuccessPass

#%%
#Add an xSuccessPass to dataframe
xSuccessPass=pass_model.apply(calculate_xSuccessfulPass, axis=1) 
pass_model = pass_model.assign(xSuccessPass=xSuccessPass)

#%%
#visualize the xSuccessfulPass across an entire pitch

#Draw the pitch
#Size of the pitch in yards (!!!)
pitchLengthX=120
pitchWidthY=80
(fig,ax) = FCPython.createPitch(pitchLengthX,pitchWidthY,'yards','gray')
