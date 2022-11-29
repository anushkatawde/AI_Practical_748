#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np


# ## R-Matrix

# In[16]:


R=np.matrix([[-1,-1,-1,-1,0,-1],
           [-1,-1,-1,0,-1,100],
           [-1,-1,-1,0,-1,1],
           [-1,0,0,-1,0,-1],
           [-1,0,0,-1,-1,100],
           [-1,0,-1,-1,0,100]])


# # Q matrix

# In[17]:


Q =np.matrix(np.zeros([6,6]))


# ## Gamma Learning Parameters

# In[18]:


gamma=0.8


# ## Initial state usually chosen at random

# In[19]:


initial_state=1


# ## This function returns all available actions in the state given an argument

# In[20]:


def available_actions(state):
    current_state_row=R[state,]
    av_act=np.where(current_state_row>=0)[1]
    return av_act


# ## Get available actions in the current state

# In[21]:


available_act=available_actions(initial_state)


# ## This function choosen at radom which to be performed within the range of all the available actions

# In[22]:


def simple_next_action(available_actions_range):
    next_action=int(np.random.choice(available_act,1))
    return next_action


# ## sample next action to be performed

# In[23]:


action=simple_next_action(available_act)


# ## This function calculates Q matrix according to the path selected and the Q learning algorithm

# In[24]:


def update(current_state,action,gamma):
    max_index=np.where(Q[action,]==np.max(Q[action,]))[1]
    
    if max_index.shape[0]>1:
        max_index=int(np.random.choice(max_index,size=1))
    else:
        max_index=int(max_index)
    max_value=Q[action,max_index]
        
    #Q learning formula        
    Q[current_state,action]=R[current_state,action] + gamma * max_value       


# ## Update Q matrix

# In[25]:


update(initial_state,action,gamma)


# ## Training                        
# Train over 10000 iterations.Re-iterate the process above.               

# In[26]:


for i in range(10000):
    current_state=np.random.randint(0,int(Q.shape[0]))
    available_act=available_actions(current_state)
    action=simple_next_action(available_act)
    update(current_state,action,gamma)


# ## Normalize the trained Q matrix

# In[27]:


print("Trained Q martix")
print(Q/np.max(Q) * 100)


# # Testing                  
# Goal state =5 best sequence path from 2->2,3,1,5

# In[ ]:


current_state =1 #Check from current_state=2
steps=[current_state]

while current_state!=5:
    next_step_index=np.where(Q[current_state,]==np.max(Q[current_state,]))[1]
    
if next_step_index.shape[0]>1:
    next_step_index=int(np.random.choice(next_step_index,size=1))
else:
    next_step_index=int(next_step_index)
    
steps.append(next_step_index)
current_state=next_step_index

#print selected sequence of steps
print("Selected path:")
print(steps)


# In[ ]:





# In[ ]:




