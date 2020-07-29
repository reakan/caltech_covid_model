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
Helper Functions for CoViD model
Updated 07/07/2020

"""


def gen_names(n):
    """
    Generates an n-list of randomly generated names (from a pre-existing rtf). DONT GO HIGHER THAN ~140
    """
    names=[]
    #maxes out at 148
    with open('sample_names.rtf') as f:
        names.append(f.readlines())
    names =list(set([x.strip().replace(' \\\'a0\\','') for x in names[0][8:]]))[:n]
    return names

def draw_locations(agent_locations,disease_states,names,ax):
    """
    agent locations is a list containing the location of each agent in the simulation
    disease states is a list of booleans describing whether or not an individual is infected (0=healthy,1=infected)
    ax is the matplotlib axes object you want to draw on
    
    returns nothing, operates in-place
    """
    #draw borders
    _=[ax.axhline(x,xmin=0,xmax=1) for x in [-2,-1,1,2]]
    _=[ax.axvline(x,ymin=0,ymax=1) for x in [-2,-1,1,2]]
    ax.set_ylim([-2,2])
    ax.set_xlim([-2,2])


    #add labels
    ax.text(1.4,-1.1,'Lab 7')
    ax.text(1.4,0.9,'Lab 4')
    ax.text(1.4,1.9,'Lab 2')
    ax.text(-.1,-1.1,'Lab 6')
    ax.text(-.1,0.9,'Chandler')
    ax.text(-.1,1.9,'Lab 1')
    ax.text(-1.6,-1.1,'Lab 5')
    ax.text(-1.6,0.9,'Lab 3')
    ax.text(-1.6,1.9,'Lab 0')

    center_coords = {'Lab 0':[-1.6,1.4],
                     'Lab 1':[-.1,1.4],
                     'Lab 2':[1.4,1.4],
                     'Lab 3':[-1.6,-0.4],
                     'Lab 4':[1.4,-0.4],
                     'Lab 5':[-1.6,-1.6],
                     'Lab 6':[-.1,-1.6],
                     'Lab 7':[1.4,-1.6],
                     'Chandler':[-.1,-0.4]}

    #iterate over people and draw each on the axis
    for person_idx in range(len(agent_locations)):
        deltax = st.uniform.rvs(scale=.25)
        deltay = st.uniform.rvs(scale=.25) #jittering points
        center = center_coords[agent_locations[person_idx]]

        if disease_states[person_idx]==0:
            color='blue'
        else:
            color='red'

        ax.plot(center[0]+deltax,center[1]+deltay,color=color,marker='*')
        ax.text(center[0]+deltax,center[1]+deltay,names[person_idx])

        
def draw_dorm(agent_locations,disease_states,ax,background_map = plt.imread('dorms.png')):
    """
    agent locations is a list containing the location of each agent in the simulation
    disease states is a list of booleans describing whether or not an individual is infected (0=healthy,1=infected)
    ax is the matplotlib axes object you want to draw on
    
    returns nothing, operates in-place
    """

    #group together states so they're not randomly shuffled
    agent_locations = [x for _,x in sorted(zip(disease_states,agent_locations))] 
    disease_states = sorted(disease_states)
    locations = ['Bechtel','Lloyd','Page','Ruddock','Chandler','Quarantine']

    alphaarray = np.linspace(0.25,1,num=3)
    colors=['blue','green','red']

    ax.imshow(background_map)


    #add labels
    ax.text(30,75,'Bechtel',size=14)
    ax.text(215,450,'Lloyd',size=14)
    ax.text(157,450,'Page',size=14)
    ax.text(230,350,'Ruddock',size=14)
    ax.text(170,340,'Chandler',size=14)
    ax.text(230,210,'Quarantine',size=14)

    #each value is a 2 list describing the ranges on x and y for that location
    coor_ranges = {
        'Bechtel': [[10,100],[5,60]],
        'Lloyd': [[200,230],[377,450]],
        'Page': [[150,200],[377,450]],
        'Ruddock': [[230,275],[360,420]],
        'Chandler': [[150,230],[340,385]],
        'Quarantine':[[225,290],[5,200]]
  
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
def gen_initstate(statefreqs=[0.48,0.3,0.2]+6*[0.02/6]+[0,0],N=10):
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
        
#################################
# Helper Functions for Simulation
#################################

def move_agent(current_time,prev_idx,current_idx,person_idx,locations,agent_locations,agent_loctmats,lunchtimes,lunchloc='Chandler'):
    """
    # Inputs: lunchtimes; a 2-list containing start and end times for lunches, 
    current_time; a counter for number of intervals since start of day,
    current_idx; a counter for number of intervals since start of simulation, 
    agent_locations; array of shape timesteps,npeople, describing agent locations in strings
    locations; list of strings containing location names. CHANDLER SHOULD ALWAYS BE LAST., 
    person_idx; index of person whose location we need to update
    
    # Outputs: current_location,agent_locations for person specified by person_idx
    """
    agent_locations_out = np.copy(agent_locations)  #so we're super duper sure nothing is overwritten
    
    
    #first we check lunchtime conditions,check if any lunchtimes have started or if we're in the middle of a lunch period
    if lunchtimes[1][0][person_idx]>=current_time and current_time>= lunchtimes[0][0][person_idx]:
        if agent_locations_out[prev_idx,person_idx] != 'Quarantine':
            agent_locations_out[current_idx,person_idx] = lunchloc #force that person to go to Chandler within their planned lunch period
            current_location = lunchloc
        else:
            agent_locations_out[current_idx,person_idx] = 'Quarantine'
            current_location = 'Quarantine'
    else:
        #call transition matrix
        tmat = agent_loctmats[person_idx]
        
        if agent_locations_out[prev_idx,person_idx]=='Quarantine':
            agent_locations_out[current_idx,person_idx] = 'Quarantine'
            return 'Quarantine',agent_locations_out
        
        prev_location = locations.index(agent_locations_out[prev_idx,person_idx])
        if prev_location == len(locations)-1:
            prev_location = np.random.choice(np.arange(len(locations[:-1])))  #in case the last location is chandler, feed in a random location so things don't break
               
        #vectorize the location
        prev_vec = np.zeros([1,len(locations[:-1])])
        prev_vec[0,prev_location] = 1
                
        current_location = np.random.choice(locations[:-1],p=np.squeeze(np.matmul(prev_vec,tmat)))
        agent_locations_out[current_idx,person_idx] = current_location
    
    return current_location,agent_locations_out
        

def transmission(prev_idx,current_idx,person_idx,locations,agent_locations,current_location,agent_states,loc_weights,lambda_,pse=None):
    """
    
    DEPRECATED
    
    Inputs: lambda_ (for person i); sensitivity parameter,
    prev_idx; index of previous timepoint (you can't just subtract 1 from the current_idx, there's some weird indexing stuff),
    current_idx; counter for number of intervals since the begining of the simulation,
    agent_locations;  array of shape timesteps,npeople, describing agent locations in strings
    current_location; string describing the current location of the agent,
    agent_states; array of shape timesteps,11,npeople
    loc_weights; set of weights to modulate exposure by location, [0,1]
    Outputs: agent_states  
    """
    agent_states_out = np.copy(agent_states)
    
    if pse==None:
        #get indices of people at agent location
        loc_indices = np.where(agent_locations[current_idx,:]==current_location)[0]

        infect_list = [np.sum(agent_states_out[current_idx,6:9,x]) for x in loc_indices]
        ninfected = np.sum(infect_list)


        if current_location=='Quarantine':  #no infection possibility if in quarantine
            pse=0
        else:
            pse = 1. - np.exp(-lambda_*loc_weights[locations.index(current_location)]*ninfected)

        if np.isnan(pse) or np.isinf(pse):  #cleaning up errors
            pse=0
            
    #create transition matrix
    tmat = state_tmat(pse)
    prev_state = np.where(agent_states_out[prev_idx,:,person_idx]==1)[0] 
    
 
    if len(prev_state)!=1:  #for when things REALLY fuck up
        print('current_idx: ',current_idx,' prev_state; ',prev_state)
       
    #transition state based on probabilities
    next_state = np.random.choice(np.arange(agent_states_out.shape[1]),p=np.squeeze(tmat[prev_state,:]))
    agent_states_out[current_idx,next_state,person_idx] = 1
    
    return agent_states_out



#####################
def cum_transmission(pse,agent_states,current_idx,person_idx,lambda_):
    """
    pse is the cumulative transimission probability that was accumulated throughout the day
    agent_states is the timestepx11xnpeople tensor that records the state of each agent
    current_idx is the current day index
    person_idx refers to the person whose state we're changing. THIS BELONGS IN A FOR LOOP
    
    returns an updated version of agent_states
    """
    #create transition matrix
    pse=pse[person_idx]
    pse = 1-np.exp(-lambda_*pse)
    prev_idx = current_idx-1
    agent_states_out = np.copy(agent_states)
    
    tmat = state_tmat(pse)
    #print('pse: ',pse)
    prev_state = np.where(agent_states[prev_idx,:,person_idx]==1)[0] 
    
 
    if len(prev_state)!=1:  #for when things REALLY fuck up
        print('current_idx: ',current_idx,' prev_state; ',prev_state)
       
    #transition state based on probabilities
    next_state = np.random.choice(np.arange(agent_states_out.shape[1]),p=np.squeeze(tmat[prev_state,:]))
    
    agent_states_out[current_idx,next_state,person_idx] = 1
    #print('ago: ',agent_states_out[current_idx,:,person_idx])
    return agent_states_out


#####################
                
def run_simulation(ndays,npeople,locations,loc_weights=None,init_locations=None,init_states=None,lunchtimes=None,testsperday=25,testdelay=1,testingpolicy='Random',quarantinelen=14,lambda_=1e3,loctmats=None):
    """
    Runs CoViD main simulation
    
    Inputs:
    ndays is an int describing the number of days to run the simulation
    npeople is an int describing the number of people to model
    locations is a list of strings containing the names of the locations
    loc_weights is a vector the same length as locations to tune exposure by location (assumed [0,1])
    init_locations is an array npeople long describing the initial location of each agent
    init_states is an array [11,npeople] that describes the state of every person at the begining of the simulation
    testsperday is an int describing the number of diagnostic tests to run per day
    testdelay is the number of days to wait until getting test results USE TESTDELAY=1 FOR INSTANT RESULTS.
    testingpolicy is the policy to take when selecting people for tests
    quarantinelen is an int describing the number of days to quarantine a person
    
    
    Outputs:
    agent_locations is an array [timesteps,npeople] describing the location of each person at each timestep
    agent_states is an array [timesteps,11,npeople] describing the state of each person in a one-hot fashion
    """
    
    # a really hacky way to retrieve test/quarantine results. counter floors at 0 so i can't specify to check indices equal to zero; ill pick up everybody. instead, we add one day to the test delay and check all indices equal to 1
    testdelay+=1
    quarantinelen+=1
    
    
    if loc_weights==None:
        #instantiating weight matrices for locations
        loc_weights = st.uniform.rvs(size=len(locations))
    
    # instantiating arrays to hold location and state
    agent_locations = np.empty(shape=[96*(ndays-1),npeople],dtype=object)
    agent_states = np.zeros(shape=[ndays,11,npeople])
    
    if init_locations==None:
        agent_locations[0,:] = np.random.choice(locations[:-1],size=npeople)
    else:
        agent_locations[0,:] = init_locations
    
    if type(init_states)==np.ndarray:
        agent_states[0,...] = init_states
    else:
        agent_states[0,...] = gen_initstate(N=npeople)

    
    #guarantee at least 1 person w/ covid is in the simulation
    if np.sum(agent_states[0,...])==0:
        agent_states[0,-3,-1] = 1 #make the last person (very) sick if no one else is
    
    
    #generate transition matrices
    if loctmats == None:
        agent_loctmats = [loc_tmat(np.random.choice(np.arange(len(locations)-1)),len(locations)-1) for _ in range(npeople)]
    else:
        agent_loctmats = loctmats
        
        
    #instantiating counters
    quarantine_list=[]  #a list to hold indices of people in quarantine
    quarantine_counter = np.zeros(shape=npeople)  #a counter to hold days since begining of quarantine
    test_counter = np.zeros(shape=npeople) #a counter to hold days since diagnostic test is run
    test_blacklist = []
    testarray = np.zeros([11,npeople])  # a vector holding the state of tested people
    prev_idx=0
    current_idx = 0
    
    
    #main simulation loop
    for day_idx in tqdm(range(1,ndays),desc='Running Simulation'):  #loop over days
        #print('Day: ',day_idx,'/',ndays,end='\r')
        
        if lunchtimes==None:  #if lunchtimes change everyday, write them in
            lunchtimes = get_lunchtimes(npeople=npeople)
        current_time=0

        #start the day in a random location
        agent_locations[(day_idx-1)*96,:] = np.random.choice(locations[:-1],size=npeople)
        cum_exposure = np.zeros(npeople)
        
        
        
        #####################################
        # Quarantine Step
        #####################################
        
        #first, we decrement the testing counter
        test_counter = np.maximum(np.zeros(npeople),test_counter-1)
        
        #then we retrieve any tests that are ready
        result_indices = np.where(test_counter==1)[0]
        
        #anyone who is found to be positive has their location changed to quarantine
        result_states = [np.where(testarray[:,x]==1)[0][0] for x in result_indices]
        pos_indices=[]
        for test_idx in range(result_indices.size):
            if result_states[test_idx] in [3,4,5,6,7,8]:  #assuming we can catch all exposed and infected people
                pos_indices.append(result_indices[test_idx])
        
        for person_idx in range(agent_states.shape[-1]):
            index = np.where(agent_states[day_idx-1,:,person_idx]==1)[0]  #index of the 1 in the one hot
            if index in [2,5,8] and person_idx not in test_blacklist:  #if person is exhibiting strong symptoms
                pos_indices.append(person_idx)
        
        
        _=[test_blacklist.remove(x) for x in result_indices]  #take everyone off the blacklist that just got their test results
        test_blacklist+=pos_indices  #but add back the people we found that were positive, no sense testing them again; THIS LINE IS SUSPECT, COMMENT ME OUT NEXT
        
        for pos_idx in pos_indices:
                quarantine_counter[pos_idx]+= quarantinelen
                agent_locations[(day_idx-1)*96,pos_idx] = 'Quarantine'
                agent_locations[current_idx,pos_idx] = 'Quarantine'
        
        #for anyone who still has time left in quarantine; shove them back in quarantine.
        quarantine_indices = np.where(quarantine_counter>1) 
        for q_idx in range(len(quarantine_indices)):
            agent_locations[(day_idx-1)*96,quarantine_indices[q_idx]] = 'Quarantine'
            agent_locations[current_idx,quarantine_indices[q_idx]] = 'Quarantine'
        #now we need to remove people from quarantine whose time is up
        q_out = np.where(quarantine_counter==1)[0]
        for q_idx in q_out:
            rand_choice = np.random.choice(locations[:-1])
            agent_locations[(day_idx-1)*96,q_idx] = rand_choice
            agent_locations[current_idx,q_idx] = rand_choice
            
        
        ###################################
        # End Quarantine Step
        ###################################
        
        #loop over the day from 9-5
        for time_idx in range(1,96): #loop over small time interval (now 5 minutes.)
            
            #so I don't have to calculate this value a ton of times
            current_idx = (day_idx-1)*96+time_idx

            for person_idx in range(npeople):  #first move everyone to their respective locations
                current_location,agent_locations = move_agent(current_time,prev_idx,current_idx,person_idx,locations,agent_locations,agent_loctmats,lunchtimes)
                
            for person_idx in range(npeople):  #then write in the cumulative exposure
                current_location = agent_locations[current_idx,person_idx]
                
                if current_location != 'Quarantine':
                    loc_indices = np.where(agent_locations[current_idx,:]==current_location)[0]
                    infect_list = [np.sum(agent_states[day_idx-1,6:9,x]) for x in loc_indices]
                    ninfected = np.sum(infect_list)
                    cum_exposure[person_idx] += loc_weights[locations.index(current_location)]*ninfected
                elif current_location == 'Quarantine':
                    cum_exposure[person_idx]+= 0. #replace 0 with community exposure rate
              
            
            
            prev_idx = current_idx
            current_time+=1
        
        #advance the state once a day
        for person_idx in range(npeople):
            agent_states = cum_transmission(cum_exposure,agent_states,day_idx,person_idx,lambda_=lambda_)
            
            
        ######################################
        # Testing Step
        ######################################
        if testsperday>=0:
            validindices = [x for x in np.arange(npeople) if x not in test_blacklist]
            
            
            if testingpolicy=='Random':
                #policy only really changes this line
                people_to_test = list(np.random.choice(validindices,size=min([testsperday,len(validindices)]),replace=False))  #we're picking people, replacement would make no sense
                #print('people_to_test Random: ',people_to_test)
                
            elif testingpolicy=='Greedy':
                #for greedy testing, we run a forecasting step; then allocate all tests to the most "at risk" individuals (most likely to be exposed or infected)
                #print('current_idx: ',current_idx)
                #print('prev_locations shape: ',agent_locations[current_idx-95:current_idx+1,:].shape)  #the 95 and +1 are for indexing
                #print('result_indices (people w/ known disease status): ',list(result_indices))
                forecast_output = forecast(50,agent_locations[current_idx-95:current_idx+1,:],locations,agent_states[day_idx,...],loc_weights,lambda_=lambda_,known_indices=list(result_indices))
                exposure_score = np.sum(forecast_output[3:9,:],axis=0)
                sortlist = list(np.flip(np.argsort(exposure_score)))
                sortlist = [x for x in sortlist if x in validindices]
                people_to_test = sortlist[:min([testsperday,len(validindices)])]
                #print('people_to_test Greedy: ',people_to_test)
                
                
            test_blacklist+=people_to_test  #if someone's waiting for their results, don't test them again
            test_counter[people_to_test] += testdelay  #  start the counter
            for person_idx in people_to_test:

                state_idx = min([np.where(agent_states[day_idx,:,person_idx]==1)[0],10])
                testarray[:,person_idx]=0 #clear the column just in case it isn't already clear
                testarray[state_idx,person_idx]=1  #record the state at this time, but don't look until the counter fully decrements
        ######################################
        # End Testing Step
        ######################################
        
        
        quarantine_counter = np.maximum(np.zeros(npeople),quarantine_counter-1)  #decrement the quarantine counter
        

        
        
    return agent_locations,agent_states
            
    
    
def dorm_simulation(ndays,loc_weights=None,init_states=None,lunchtimes=None,testsperday=0,testdelay=2,testingpolicy='Random',quarantinelen=14,lambda_=1e9,loctmats=None,lunchloc='Chandler',mini=False):
    """
    Runs CoViD main simulation
    
    Inputs:
    ndays is an int describing the number of days to run the simulation
    npeople is an int describing the number of people to model
    locations is a list of strings containing the names of the locations
    loc_weights is a vector the same length as locations to tune exposure by location (assumed [0,1])
    init_locations is an array npeople long describing the initial location of each agent
    init_states is an array [11,npeople] that describes the state of every person at the begining of the simulation
    testsperday is an int describing the number of diagnostic tests to run per day
    testdelay is the number of days to wait until getting test results
    testingpolicy is the policy to take when selecting people for tests
    quarantinelen is an int describing the number of days to quarantine a person
    
    
    Outputs:
    agent_locations is an array [timesteps,npeople] describing the location of each person at each timestep
    agent_states is an array [timesteps,11,npeople] describing the state of each person in a one-hot fashion
    """
    
    # a really hacky way to retrieve test/quarantine results. counter floors at 0 so i can't specify to check indices equal to zero; ill pick up everybody. instead, we add one day to the test delay and check all indices equal to 1
    testdelay+=1
    quarantinelen+=1
    locations = ['Bechtel','Lloyd','Page','Ruddock','Chandler']
    
    if mini:
        homelocs = 30*['Bechtel']+10*['Lloyd']+10*['Page']+10*['Ruddock']
    else:
        homelocs = 150*['Bechtel']+50*['Lloyd']+50*['Page']+50*['Ruddock']
        
    npeople=len(homelocs)
    
    #if loc_weights==None:
        #instantiating weight matrices for locations
        #loc_weights = st.uniform.rvs(size=len(locations))
        
    loc_weights = np.ones(len(locations))
        
    # instantiating arrays to hold location and state
    agent_locations = np.empty(shape=[96*(ndays-1),npeople],dtype=object)
    agent_states = np.zeros(shape=[ndays,11,npeople])
    

    agent_locations[0,:] = homelocs
    
#     if init_states==None:
#         agent_states[0,...] = gen_initstate(N=npeople)
#     else:
#         agent_states[0,...] = init_states

    agent_states[0,...] = gen_initstate(N=npeople)
    #generate transition matrices
    agent_loctmats = [loc_tmat(locations.index(x),len(locations)-1,ff=10000) for x in homelocs]
    
    #guarantee at least 1 person w/ covid is in the simulation
    if np.sum(agent_states[0,...])==0:
        agent_states[0,-3,-1] = 1 #make the last person (very) sick if no one else is
    
    
    #main simulation loop
    quarantine_list=[]  #a list to hold indices of people in quarantine
    quarantine_counter = np.zeros(shape=npeople)  #a counter to hold days since begining of quarantine
    test_counter = np.zeros(shape=npeople) #a counter to hold days since diagnostic test is run
    test_blacklist = []
    testarray = np.zeros([11,npeople])  # a vector holding the state of tested people
    
    prev_idx=0
    current_idx = 0
    for day_idx in tqdm(range(1,ndays),desc='Running Simulation'):  #loop over days
        print('Day: ',day_idx,'/',ndays,end='\r')
        
        if lunchtimes==None:  #if lunchtimes change everyday, write them in
            lunchtimes = get_lunchtimes(npeople=npeople)
        current_time=0

        #start the day in home location
        agent_locations[(day_idx-1)*96,:] = homelocs#np.random.choice(locations[:-1],size=npeople)

#         if day_idx!=0:
#             agent_states[day_idx,:] = agent_states[prev_idx,:] #carrying over the state isn't valid if it's day 0
        cum_exposure = np.zeros(npeople)
        
        
        
        #####################################
        # Quarantine Step
        #####################################
        
        #first, we decrement the testing counter
        #test_counter = np.maximum(np.zeros(npeople),test_counter-1)
        
        #then we retrieve any tests that are ready
        #result_indices = np.where(test_counter==1)[0]
        
        #anyone who is found to be positive has their location changed to quarantine
        #result_states = [np.where(testarray[:,x]==1)[0][0] for x in result_indices]
        pos_indices=[]
        
        for person_idx in range(agent_states.shape[-1]):
            index = np.where(agent_states[day_idx-1,:,person_idx]==1)[0]  #index of the 1 in the one hot
            if index in [2,5,8]:  #if person is exhibiting strong symptoms
                pos_indices.append(person_idx)
        #for test_idx in range(result_indices.size):
        #    if result_states[test_idx] in [3,4,5,6,7,9]:  #assuming we can catch all exposed and infected people
        #        pos_indices.append(result_indices[test_idx])
        
        #_=[test_blacklist.remove(x) for x in result_indices]  #take everyone off the blacklist that just got their test results
        #test_blacklist+=pos_indices  #but add the people we found that were positive, no sense testing them again
        
        
        for pos_idx in pos_indices:
                quarantine_counter[pos_idx]+= quarantinelen
                agent_locations[current_idx,pos_idx] = 'Quarantine'
                agent_locations[(day_idx-1)*96,pos_idx] = 'Quarantine'
                
        quarantine_indices = np.where(quarantine_counter>1) 
        for q_idx in range(len(quarantine_indices)):
            agent_locations[(day_idx-1)*96,quarantine_indices[q_idx]] = 'Quarantine'  #quarantine at the begining of the day
            agent_locations[current_idx,quarantine_indices[q_idx]] = 'Quarantine'  #also prewrite the quarantine into the (previous? current_idx isn't updated at this step...a little fishy) index so move_agent doesn't shuffle at the begining of the day
                
                
        #now we need to remove people from quarantine whose time is up
        q_out = np.where(quarantine_counter==1)[0]
        for q_idx in q_out:
            #rand_choice = np.random.choice(locations[:-1])
            rand_choice = homelocs[q_idx]  #not really random, but i don't want to rewrite code.
            agent_locations[current_idx,q_idx] = rand_choice
            agent_locations[(day_idx-1)*96,q_idx] = rand_choice
        
        ###################################
        # End Quarantine Step
        ###################################
        
        
        
        
        for time_idx in range(1,96): #loop over small time interval (now 5 minutes.)
            #so I don't have to calculate this value a ton of times
            current_idx = (day_idx-1)*96+time_idx

            for person_idx in range(npeople):  #first move everyone to their respective locations
                if lunchloc!='Chandler':
                    current_location,agent_locations = move_agent(current_time,prev_idx,current_idx,person_idx,locations,agent_locations,agent_loctmats,lunchtimes,lunchloc=homelocs[person_idx])
                elif lunchloc=='Chandler':
                    current_location,agent_locations = move_agent(current_time,prev_idx,current_idx,person_idx,locations,agent_locations,agent_loctmats,lunchtimes,lunchloc='Chandler')
                
            for person_idx in range(npeople):  #then write in the cumulative exposure
                current_location = agent_locations[current_idx,person_idx]
                
                if current_location != 'Quarantine':
                    loc_indices = np.where(agent_locations[current_idx,:]==current_location)[0]
                    infect_list = [np.sum(agent_states[day_idx-1,6:9,x]) for x in loc_indices]
                    ninfected = np.sum(infect_list)
                    cum_exposure[person_idx] += loc_weights[locations.index(current_location)]*ninfected
                elif current_location == 'Quarantine':
                    cum_exposure[person_idx]+= 0. #replace 0 with community exposure rate
              
            
            
            prev_idx = current_idx
            current_time+=1
        
        for person_idx in range(npeople):
            agent_states = cum_transmission(cum_exposure,agent_states,day_idx,person_idx,lambda_=lambda_)
            
            
#         ######################################
#         # Testing Step
#         ######################################
#         if testsperday>=0:
            
#             if testingpolicy=='Random':
#                 validindices = [x for x in np.arange(npeople) if x not in test_blacklist]
                
#                 #policy only really changes this line
#                 people_to_test = list(np.random.choice(validindices,size=min([testsperday,len(validindices)]),replace=False))  #we're picking people, replacement would make no sense
                
#                 test_blacklist+=people_to_test  #if someone's waiting for their results, don't test them again
#                 test_counter[people_to_test] += testdelay  #  start the counter
#                 for person_idx in people_to_test:
#                     #print(agent_states[current_idx,:,person_idx])
#                     #print('one-hot indices',np.where(agent_states[day_idx,:,person_idx]==1)[0])
#                     state_idx = min([np.where(agent_states[day_idx,:,person_idx]==1)[0],10])
#                     testarray[:,person_idx]=0 #clear the column just in case it isn't already clear
#                     testarray[state_idx,person_idx]=1  #record the state at this time, but don't look until the counter fully decrements
                    
#         quarantine_counter = np.maximum(np.zeros(npeople),quarantine_counter-1)  #decrement the quarantine counter
        
        ######################################
        # End Testing Step
        ######################################
        
        
        
    return agent_locations,agent_states
    
    
    
    

def forecast(ntrials,prev_locations,locations,init_state,loc_weights,known_indices=[],lambda_=1e-2):
    """ 
    Inputs:
    ntrials is an int describing how many monte carlo trials to run. works well with 100 right now. might need to tweak based on number of people in simulation
    prev_locations is an [96,npeople] array describing the locations of each person during the forecast period
    init_state is an [11,npeople] matrix that's encoded with the possible states of each person (we assume symptoms can be visualized.) 1 in a column denotes possible state
    loc_weights is a vector nlocations long that describes the transmission multiplier at that location
    
    Outputs:
    returnarray an [11,npeople] array that describes the probability of a person occupying a particular state at the end of the forecast period
    """
    
    #we assume that the symptomatic information is visible, but we can't see disease status
    init_state_full = np.copy(init_state)
    unknown_indices = [x  for x in np.arange(prev_locations.shape[-1]) if x not in known_indices]

    for person_idx in unknown_indices:  #if we don't know what they have, use full info from symptoms. Instead of a one-hot, we'll get a 3-hot array out

         #asymptomatic case
        if np.where(init_state_full[:,person_idx]==1)[0][0] in [0,3,6,9]:
            init_state_full[[0,3,6,9],person_idx]=1
    
         #semi-strong symptom case
        if np.where(init_state_full[:,person_idx]==1)[0][0] in [1,4,7]:
            init_state_full[[1,4,7],person_idx]=1
    
        #super-strong symptom case
        if np.where(init_state_full[:,person_idx]==1)[0][0] in [2,5,8]:
            init_state_full[[2,5,8],person_idx]=1

    returnarray = np.zeros(shape=[11,prev_locations.shape[-1]])
    
    for mc_idx in tqdm(range(ntrials)):
        #print('Forecasting...MC Trial: ',mc_idx,'/',ntrials,end='\r')
        
        # instantiating arrays to hold location and state
        agent_states_forecast = np.zeros(shape=[2,11,prev_locations.shape[-1]])
    
        #sample an initial state from init_state_full
        for person_idx in range(prev_locations.shape[-1]):
            agent_states_forecast[0,np.random.choice(np.where(init_state_full[:,person_idx]==1)[0]),person_idx] = 1

        #main simulation loop
        cum_exposure = np.zeros(prev_locations.shape[-1])
        prev_idx=0
        for time_idx in range(1,96): #loop over small time interval (now 5 minutes.)
            for person_idx in range(prev_locations.shape[-1]):
                current_location = prev_locations[time_idx,person_idx] 
                if current_location != 'Quarantine':
                    loc_indices = np.where(prev_locations[time_idx,:]==current_location)[0]
                    infect_list = [np.sum(agent_states_forecast[0,3:9,x]) for x in loc_indices]
                    ninfected = np.sum(infect_list)
                    cum_exposure[person_idx] += ninfected*loc_weights[locations.index(current_location)]
                elif current_location == 'Quarantine':
                    cum_exposure[person_idx]+= 0. #replace 0 with community exposure rate
        
        for person_idx in range(prev_locations.shape[-1]):
            agent_states_forecast = cum_transmission(cum_exposure,agent_states_forecast,1,person_idx,lambda_=lambda_)
            
        returnarray+=agent_states_forecast[-1,...]
    returnarray /= ntrials
    return returnarray
    
    

    
    
    
