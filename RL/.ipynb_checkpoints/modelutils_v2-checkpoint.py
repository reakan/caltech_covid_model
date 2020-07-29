#!/usr/bin/env python3

#reidakana

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats as st
import itertools as it
from tqdm import tqdm_notebook as tqdm
import random

"""
Helper Functions for CoViD model. 

UPDATED v2; slightly more elegant implementation, made for easy integration w/ tensorflow and d3. Use older version to manually test 'exotic' policies.

Updated 07/27/2020

"""
        
        
def draw_campus(agent_locations,disease_states,ax,background_map = plt.imread('campus_map.png')):
    """
    agent locations is a list containing the location of each agent in the simulation
    disease states is a list of booleans describing whether or not an individual is infected (0=healthy,1=infected)
    ax is the matplotlib axes object you want to draw on
    
    returns nothing, operates in-place
    """

    #group together states so they're not randomly shuffled
    agent_locations = [x for _,x in sorted(zip(disease_states,agent_locations))] 
    disease_states = sorted(disease_states)

    locations=['Broad','Moore','Watson','Beckman','Noyes','Braun','Schlinger','Kerckhoff','Spalding','Chandler','Quarantine']
    alphaarray = np.linspace(0.25,1,num=3)
    colors=['blue','green','red']

    ax.imshow(background_map)


    #add labels
    ax.text(17,95,'Broad',size=14)
    ax.text(145,95,'Moore',size=14)
    ax.text(157,170,'Watson',size=14)
    ax.text(45,120,'Beckman',size=14)
    ax.text(47,168,'Noyes',size=14)
    ax.text(17,185,'Braun',size=14)
    ax.text(65,185,'Schlinger',size=14)
    ax.text(25,250,'Kerckhoff',size=14)
    ax.text(147,225,'Spalding',size=14)
    ax.text(205,223,'Chandler',size=14)
    ax.text(25,10,'Quarantine',size=14)

    #each value is a 2 list describing the ranges on x and y for that location
    coor_ranges = {
        'Broad': [[10,45],[100,125]],
        'Moore': [[130,175],[100,125]],
        'Watson': [[150,190],[125,160]],
        'Beckman': [[45,75],[125,165]],
        'Noyes': [[40,70],[170,205]],
        'Braun':[[13,35],[187,227]],
        'Schlinger':[[60,90],[187,215]],
        'Kerckhoff':[[10,75],[253,270]],
        'Spalding':[[142,180],[227,240]],
        'Quarantine':[[0,125],[0,75]],
        'Chandler':[[202,242],[225,242]]   
    }

    people_counter = np.zeros(len(coor_ranges.keys())).astype(np.int)

    gridlen = np.ceil(np.sqrt(len(agent_locations))).astype(np.int) #create enough gridpoints such that everyone has a space if they mob a single location (chandler)

    #iterate over people and draw each on the axis
    for person_idx in range(len(agent_locations)):

        color_idx = int(disease_states[person_idx]//3)
        alpha_idx = int(disease_states[person_idx] % 3)
        if color_idx<=2:
            color=colors[color_idx]
            alpha=alphaarray[alpha_idx]
        elif disease_states[person_idx]==9:
            color='magenta'
            alpha=1
        elif disease_states[person_idx]==10:
            color='black'
            alpha=1

        people_counter[locations.index(agent_locations[person_idx])] += 1

        #draw a grid over the current location and place agent at specified spot
        current_ranges = coor_ranges[agent_locations[person_idx]]
        xx,yy = np.meshgrid(np.linspace(current_ranges[0][0],current_ranges[0][1],num=gridlen),np.linspace(current_ranges[1][0],current_ranges[1][1],num=gridlen))

        #we need to convert the raw counts into row,col indices to get the coordinates of interest
        pal = people_counter[locations.index(agent_locations[person_idx])]-1 #again, -1 is for indexing. pal stands for people at location

        ax.plot(xx[pal//gridlen,pal%gridlen],yy[pal//gridlen,pal%gridlen],color=color,alpha=alpha,marker='*',markersize=10,markeredgecolor='black')

           
def loc_tmat(homeloc,totallocs,ff=100):
    """
    generates a right stochastic matrix (transition matrix)
    homeloc is the location the person "hovers" at and returns to. totallocs is the number of locations open to the agents
    ff is a fudge factor
    
    returns a matrix totallocs x totallocs
    """

    #adding 1 to diagonals off the bat to bias staying at a location
    returnmat = np.eye(totallocs)*ff

    #adding a small amount to represent transitions between locations. We're sampling from a half normal.
    returnmat+= np.reshape(np.abs(st.norm.rvs(size=[totallocs*totallocs],scale=0.1)),[totallocs,totallocs])

    returnmat[:,homeloc]+= ff #biasing movement towards home location
    returnmat[homeloc,homeloc]+= ff #making sure we stay in home location if we get there
    returnmat = returnmat / returnmat.sum(axis=1)[:,None]  #normalization

    return returnmat
        

def get_lunchtimes(lunch_start=36,mean_time=12,npeople=20):
    """
    lunch_start is a scalar that describes the number of 5-minute intervals from 9 o clock that people will begin their lunch break (in 5-minute interval units; typically 36)
    mean_time is the average length of a lunchbreak, held to be 12 5-minute intervals or 60 minutes. units are 5-minute intervals
    npeople describes the number of people to generate lunchtimes for
    returns a 2-list. the first list contains the lunch start times (in 5-minute interval units) and the second list contains the end times
    """
    start_times = [lunch_start + st.poisson.rvs(mu=10,size=npeople)]  #noising our start times with a poisson
    duration = [st.poisson.rvs(mu=mean_time,size=npeople)]
    end_times = [x+y for x,y in zip(start_times,duration)]
    return [start_times,end_times]

#state will be an 11 x N matrix
def gen_initstate(statefreqs=[0.3,0.3,0.2]+6*[0.2/6]+[0,0],N=10):
    """
    statefreqs is an 11-list describing the fraction of N occupying a particular state (healthy asymptomatic, healthy symptomatic, etc...)
    N is the total number of people AND A MULTIPLE OF 60
    returns an 11xN sparse matrix describing the state of each individual (by columns)
    
    """
    returnmat = np.zeros([11,N])
    returnmat[np.random.choice(np.arange(11),p=statefreqs,size=N),np.arange(N)]=1 
    return returnmat
         
def state_tmat(pse=0.0000001):
    """
    pse is a float describing the probability someone in a non-infected state will transition to an infected state (per time interval)
    returns an 11x11 transition matrix
    
    default pse value should reflect community rate of infection (CHECK ME!!!!!!!!)
    """
    
    outarray = np.eye(11) #takes care of the last two rows from the get-go
    outarray[0,:]=[0.99*(1-pse),0.01*(1-pse),0,pse,0,0,0,0,0,0,0]
    
    outarray[1,:] = [0.2*(1-pse),0.799*(1-pse),0.001*(1-pse),0,pse,0,0,0,0,0,0]
    outarray[2,:] = [0,0.3*(1-pse),0.7*(1-pse),0,0,pse,0,0,0,0,0]
    
    
    outarray[3,:] = [0,0,0,1-(1/5),0,0,0.94*(1/5),0.05*(1/5),0.01*(1/5),0,0]
    
    outarray[4,:] = [0,0,0,0,1-(1/5),0,0,0.9*(1/5),0.1*(1/5),0,0]
    outarray[5,:] = [0,0,0,0,0,1-(1/5),0,0,1/5,0,0]
    outarray[6,:] = [0,0,0,0,0,0,0.9*(1-1/9),0.1*(1-1/9),0,1/9,0]
    outarray[7,:] = [0,0,0,0,0,0,0,0.9*(1-1/9),0.1*(1-1/9),1/9,0]
    
    outarray[8,:] = [0,0,0,0,0,0,0,0,1-1/9,0.95*(1/9),0.05*(1/9)]
    return outarray        


def gen_plotmat(statelist):
    """
    input is a list of states n long. We assume the first index is the earliest time.
    returns an n x 5 matrix describing the time series of the simulation (time on first dim). The 5 cols refer to SEIRD

    list of output arrays will go into gen_conf_int() function for easy plotting
    """
    returnmat = np.zeros([len(statelist),5])

    for state_idx in range(len(statelist)):
        current_state = statelist[state_idx]

        returnmat[state_idx,0] = np.sum(current_state.disease_states[:3,:])  #sum over susceptible people
        returnmat[state_idx,1] = np.sum(current_state.disease_states[3:6,:])  #sum over exposed people
        returnmat[state_idx,2] = np.sum(current_state.disease_states[6:9,:])  #sum over infected people
        returnmat[state_idx,3] = np.sum(current_state.disease_states[9,:])  #recovered people
        returnmat[state_idx,4] = np.sum(current_state.disease_states[10,:])  #recovered people

    return returnmat

def gen_tseries_list(init_state,tsteps,ntrials,action):
    """
    generates a list of timeseries, seeded with initial state init_state. 
    runs ntrials to generate confidence intervals, runs a trial tsteps long.
    """
    tseries_list = []
    
    if action==None:
        action = np.zeros(init_state.npeople)
    
    for _ in tqdm(range(ntrials)):
        print('Trial: ',_,end='\r')
        states = [init_state]
        for time_idx in range(tsteps-1):  #sample a time series based on an initial state
            states.append(states[-1].update_state(action))

        tseries_list.append(gen_plotmat(states))
    return tseries_list

def gen_percentiles(tseries_list):
    """
    tseries_list is a list of ntimepoints x 5 matrices that describes a timeseries trial used to generate confidence intervals
    
    generates a 6 x ntimepoints x 5 array describing confidence intervals for each SEIRD at all timepoints evaluated.
    assumes the confidence intervals we want are 10,20,30,70,80,90
    
    
    """
    outarray = np.zeros(shape=[6]+list(tseries_list[0].shape))  #represents 10,20,30,70,80,90 percentiles

    concat_plotarray = np.concatenate([np.expand_dims(x,axis=0) for x in tseries_list],axis=0)

    outarray[0,...] = np.percentile(concat_plotarray,10,axis=0)
    outarray[1,...] = np.percentile(concat_plotarray,20,axis=0)
    outarray[2,...] = np.percentile(concat_plotarray,30,axis=0)
    outarray[3,...] = np.percentile(concat_plotarray,90,axis=0)
    outarray[4,...] = np.percentile(concat_plotarray,80,axis=0)
    outarray[5,...] = np.percentile(concat_plotarray,70,axis=0)

    return outarray

def plot_percentiles(percentiles,tseries_list,ax=None):
    """
    percentiles is a 6 x ntimepoints x 5 array describing the percentiles of SEIRD for all timepoints

    if ax is none, returns a pyplot axis object with percentiles plotted.
    else, plots on the provided axis
    """
    alphalist = np.linspace(0.1,.3,3)
    colorlist=['blue','green','red','magenta','black']
    time = np.arange(percentiles.shape[1])
    npeople = np.sum(tseries_list[0][0,:])
    percentiles /= npeople
    #purely used for mean calculation
    concat_plotarray = np.concatenate([np.expand_dims(x/npeople,axis=0) for x in tseries_list],axis=0)

    if ax==None:
        fig,ax = plt.subplots(figsize=(10,10))  #change figsize for different monitors
        ax_is_none = True
    else:
        ax_is_none = False

    for plot_idx in range(len(percentiles)//2):

        for seir_idx in range(5):
            ax.fill_between(time,percentiles[plot_idx,:,seir_idx],percentiles[plot_idx+2,:,seir_idx],color=colorlist[seir_idx],alpha=alphalist[plot_idx])

            if plot_idx==1:
                ax.plot(time,np.mean(concat_plotarray[:,:,seir_idx],axis=0),color=colorlist[seir_idx])
    ax.legend(['Susceptible','Exposed','Infected','Recovered','Dead'])
    ax.set_xlabel('time (Days)')
    ax.set_ylabel('Fraction of Population')
    
    if ax_is_none:
        return fig,ax
    else:  #we assume the supplied axis already belongs to a pre-existing figure.
        return ax


#########################################################################
# Class definition for simulation...should probably put in separate file.
#########################################################################

class state:
    
    def __init__(self,home_locations,lambda_=5e-3,ff=None,quarantine=True):
        """
        instantiates a state object. home_locations is an nlocations x npeople matrix describing the location that each person hangs around. we assume everyone eats lunch at chandler.
        
        """
        
        #instantiate all class attributes
        
        if type(ff)==list:
            self.ff = ff
        else:
            self.ff = home_locations.shape[-1]*[100]
        
        self.quarantine = quarantine #boolean describing whether or not we're quarantining kids (CHANGE TO QUARANTINELEN)
        self.lambda_ = lambda_
        self.npeople = home_locations.shape[-1]
        
        #instantiate counters to hold number of tests and days left in quarantine
        self.home_locations = home_locations
        self.test_counter = np.zeros(home_locations.shape[-1])
        self.quarantine_counter = np.zeros(home_locations.shape[-1])
        
        self.exposures = np.zeros(shape=self.npeople)
        
        #instantiate disease states; 11xN; represents the disease_state of each person at the begining of the day
        self.disease_states = gen_initstate(N=home_locations.shape[-1])
        
        #instantiate a list holding transition matrices for location Markov Chain
        self.loc_tmats = [loc_tmat(np.where(home_locations[:,x]==1)[0],home_locations.shape[0],self.ff[x]) for x in range(home_locations.shape[-1])]
        
        #instantiate a location tensor; purely a placeholder when instantiated for the first time. 96 x nlocations x npeople
        self.locations = np.zeros([96,home_locations.shape[0]+2,home_locations.shape[-1]])  #adding two "columns" for Chandler and Quarantine.
        self.locations = self.move_agents() #instantiate a list of visited locations for each agent
        
        #finally, instantiate an action vector to hold indices of individuals for whom we know the disease state
        self.action = np.zeros(shape=self.npeople)
    
    #onto the methods....
    def move_agents(self):
        """
        generates a new location matrix based on loc_tmats,quarantine_counter,
            
        Chandler is always index -2, Quarantine is index -1
            
        return_shape 96 x nlocations x npeople
        """
        returnmat = np.zeros(self.locations.shape)
        lunchtimes = get_lunchtimes(npeople=self.npeople)

        #write initial locations
        returnmat[0,:-2,:] = self.home_locations
        
        #now on to the main loop
        for time_idx in range(1,96):
            for person_idx in range(self.npeople):
                locmat = self.loc_tmats[person_idx]
                returnmat[time_idx,np.random.choice(np.arange(self.home_locations.shape[0]),p=np.squeeze(locmat[np.where(returnmat[time_idx-1,:,person_idx]==1)[0],:])),person_idx]=1.
        
        #write lunch into the returnmatrix
        for x in range(len(lunchtimes[0][0])):
            returnmat[lunchtimes[0][0][x]:lunchtimes[1][0][x],:,x] = 0 #overwriting all locations in lunch, we need to erase first
            returnmat[lunchtimes[0][0][x]:lunchtimes[1][0][x],-2,x] = 1.  #middle 0 index is because the arrays come packed in lists...not the best idea admittedly.
        
        #finally, do exclusion by quarantine
        quarantine_indices = np.where(self.quarantine_counter!=0)[0]  #grab everyone in quarantine
        returnmat[:,:,quarantine_indices]=0  #erase their locations
        returnmat[:,-1,quarantine_indices]=1.  #and shove them in quarantine
        
        return returnmat
        
    def get_exposures(self):
        """
        uses locations, disease_state and quarantine_counter to calculate the exposures for each person.
            
        returns a vector npeople long
        """
        returnvec = np.zeros(self.npeople)
        
        infected_people = list(np.where(self.disease_states[6,:]==1.)[0])
        infected_people += list(np.where(self.disease_states[7,:]==1.)[0])
        infected_people += list(np.where(self.disease_states[8,:]==1.)[0])
        
        #main loop
        for person_idx in range(self.npeople):
            for time_idx in range(96):
                current_location = np.where(self.locations[time_idx,:,person_idx]==1.)[0][0]
                
                people_at_location = list(np.where(self.locations[time_idx,current_location,:]==1.)[0])
                
                returnvec[person_idx]+= len([x for x in people_at_location if x in infected_people])
                
            #in case the person is in quarantine, set their exposure to 0
            if current_location==self.locations.shape[1]-1 or self.quarantine_counter[person_idx]>0 and self.quarantine:  #second condition is a failsafe; i'm checking if the quarantining procedure works.
                returnvec[person_idx] = 0
        return returnvec
    
    def update_disease_state(self):
        """
        updates disease state; REQUIRES CALCULATED EXPOSURES
        returns disease_states at t+1
        """
        returnmat = np.zeros([11,self.npeople])

        pse = [1. for _ in range(self.npeople)]-np.exp(self.lambda_*(-self.exposures))
        
        for person_idx in range(self.npeople):
            tmat = state_tmat(pse[person_idx])
            returnmat[np.random.choice(np.arange(11),p=np.squeeze(tmat[np.where(self.disease_states[:,person_idx]==1)[0][0],:])),person_idx]=1.
        return returnmat
    
    def update_state(self,action):
        """
        action is a vector npeople long, with ones on people that we're going to test
        
        returns another instance of state, at t+1
        """
        #instantiate output object
        output_state = state(self.home_locations)
        self.action = action  #write in the action vector to reflect disease_states we know
        output_state.lambda_=self.lambda_
        
        #record the action in test_counter
        self.test_counter+=action
        output_state.test_counter = self.test_counter
        
        #look at results of test
        test_indices = np.where(action==1)[0]
        for test_idx in test_indices:
            if np.sum(self.disease_states[3:9,test_idx])>0 and self.quarantine:  #check if the person is infected or exposed
                self.quarantine_counter[test_idx]+=15. #it's 15 because we're going to subtract one at the end
        self.locations = self.move_agents()
        
        self.exposures = self.get_exposures()
        
        output_state.disease_states = self.update_disease_state()
        
        self.quarantine_counter += -1
        self.quarantine_counter = np.maximum(np.zeros(self.npeople),self.quarantine_counter)  #don't let people quarantine themselves negative days
        
        output_state.quarantine_counter = self.quarantine_counter
        
        return output_state
    
    def forecast(self,ntrials=1000):
        """
        requires that action and locations already be defined
        """
        returnmat = np.zeros(shape=[11,self.npeople])
        
        obs_state = self.get_observable_state()
        #create a new state
        for trial_idx in range(ntrials):
            print('Forecasting trial: ',trial_idx,'/',ntrials,end='\r')
            #sample an initial disease_state from partially-observable disease state
            sampled_state = np.zeros([11,self.npeople])
            for person_idx in range(self.npeople):
                sampled_state[np.random.choice(np.where(obs_state[:,person_idx]==1)[0]),person_idx]=1.
            
            
            #instantiate new state instance so we don't screw this one up.
            temp_state = state(self.home_locations,lambda_=self.lambda_,ff=self.ff,quarantine=self.quarantine)
            temp_state.locations = self.locations  #write in the locations as they're already known
            temp_state.disease_states = sampled_state
            temp_state.exposures = temp_state.get_exposures()
            returnmat+=temp_state.update_disease_state()
        
        return returnmat/ntrials
    
    ####################
    
    def forecast_v2(self,ntrials=1000):
        """
        returns a 2 list 
        first item in list is a list of partially-filled disease_state transition matrices.
        second item in list is the actual disease_state transition matrices
        """
        returnlist =  [np.zeros([11,11]) for _ in range(self.npeople)]
        
        obs_state = self.get_observable_state()
        sampled_state_history = np.zeros(obs_state.shape)
        
        for mc_idx in range(ntrials):
            print('Trial ',mc_idx,'/',ntrials-1,end='\r')
            #first, we need to sample initial states for each individual based on their symptom status and possible test results
            sampled_state = np.zeros([11,self.npeople])
            
            for person_idx in range(self.npeople):
                state_choice = np.random.choice(np.where(obs_state[:,person_idx]==1)[0])
                sampled_state[state_choice,person_idx]=1.
                sampled_state_history[state_choice,person_idx]+=1.

            #instantiate new state instance so we don't screw this one up.
            temp_state = state(self.home_locations,lambda_=self.lambda_,ff=self.ff,quarantine=self.quarantine)
            temp_state.locations = np.copy(self.locations)  #write in the locations as they're already known
            temp_state.disease_states = np.copy(sampled_state)
            temp_state.exposures = temp_state.get_exposures()
            next_state=temp_state.update_disease_state()
            
            #now we need to write in the tally's for each person's transition matrix
            for person_idx in range(self.npeople):
                returnlist[person_idx][np.where(sampled_state[:,person_idx]==1)[0][0],np.where(next_state[:,person_idx]==1)[0][0]]+=1
        
        #now we need to return the actual txn matrices
        #again, start by declaring a new state so we don't screw up the one we're currently working with.
        temp_state = state(self.home_locations,lambda_=self.lambda_,ff=self.ff,quarantine=self.quarantine)
        temp_state.locations = self.locations  #write in the locations as they're already known
        temp_state.disease_states = self.disease_states
        temp_state.exposures = temp_state.get_exposures()
        
        temp_state_pse = self.npeople*[1.]-np.exp(self.lambda_*(-temp_state.exposures))
        actual_txn_matrices = []
        for person_idx in range(self.npeople):
            actual_txn_matrices.append(state_tmat(temp_state_pse[person_idx]))
        
        return [returnlist,actual_txn_matrices]
        
            
    ####################
    
    
    def get_observable_state(self):
        """
        returns an 11 x npeople array with 1's on potential disease states per each person. Requires action be specified by current state.
        """
        returnmat = np.zeros(shape=[11,self.npeople])
        
        for person_idx in range(self.npeople):
            if np.where(self.disease_states[:,person_idx]==1.)[0][0] in [0,3,6,9] and self.action[person_idx]!= 1.:
                returnmat[[0,3,6,9],person_idx]=1.
            elif np.where(self.disease_states[:,person_idx]==1.)[0][0] in [1,4,7] and self.action[person_idx]!= 1.:
                returnmat[[1,4,7],person_idx]=1.
            elif np.where(self.disease_states[:,person_idx]==1.)[0][0] in [2,5,8] and self.action[person_idx]!= 1.:
                returnmat[[2,5,8],person_idx]=1.
            elif np.where(self.disease_states[:,person_idx]==1.)[0][0]==10:
                returnmat[10,person_idx]=1.
            elif self.action[person_idx]==1:
                returnmat[:,person_idx] = self.disease_states[:,person_idx]
        return returnmat
    
    def greedy_action(self,ntests):
        """
        generates a vector npeople long with ntests ones describing the people to test
        follows a greedy algorithm; runs forecast and grabs the people most likely to be exposed or infected at time t+1
        """
        output_vector = np.zeros(shape=self.npeople)
        
        forecast_output = self.forecast()
        exposure_score = np.sum(forecast_output[3:9,:],axis=0)
        ranked_indices = np.flip(np.argsort(exposure_score))[:ntests]
        
        for test_idx in list(ranked_indices):
            output_vector[test_idx]=1.
        #print('greedy action vector: ',np.where(output_vector==1)[0])
        return output_vector
    
    def random_action(self,ntests):
        """
        generates a vector npeople long with ntests ones describing the people to test
        follows a random policy. people are selected without regard.
        """
        returnvec = np.zeros(self.npeople)
        returnvec[:ntests]=1.
        np.random.shuffle(returnvec)
        #print('random action vector: ',np.where(returnvec==1)[0])
        return returnvec
        
    def reward(self):
        """
        returns reward associated with disease states at the end of the day after taking the action specified by the current state.
        """
        
        nsusceptible = np.sum(self.disease_states[0:3,:])
        nexposed = np.sum(self.disease_states[3:6,:])
        ninfected = np.sum(self.disease_states[6:9,:])
        ndead = np.sum(self.disease_states[10,:])
        
        test_coefficient = np.sum(self.test_counter)
        
        return nsusceptible/(np.sum(self.test_counter)+nexposed+ninfected+ndead)
    
    
    
    def flatten(self):
        """
        flattens locations,disease_states,?test_counter? for input into the nn
        
        #we need to gather the (partially-observable) disease state, agent locations, lambda_, test_counts into a fat vector
        """
        returnvec = np.concatenate([self.get_observable_state().flatten(),self.locations.flatten(),self.lambda_.flatten(),self.test_counter.flatten()])
        return returnvec
    
        

    
    
    
    



    
    
    
