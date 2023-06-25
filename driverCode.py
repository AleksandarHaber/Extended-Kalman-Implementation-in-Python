# -*- coding: utf-8 -*-
"""
Author Aleksandar Haber

Python Implementation of the extended Kalman filter
Date: June 2023

This is a driver code file that demonstrates how to use the extended Kalman filter 
implemented in the file ExtendedKalmanFilter.py

We use a pendulum example to test the extended Kalman filter
The webpage tutorials accompanying the developed code are given here:
    
Part 1:
https://aleksandarhaber.com/extended-kalman-filter-tutorial-with-disciplined-python-codes/

Part 2:
https://aleksandarhaber.com/extended-kalman-filter-tutorial-with-example-and-disciplined-python-codes-part-ii-python-codes/

Author: Aleksandar Haber 
Last Revision: June, 2023
    

"""

import numpy as np
import matplotlib.pyplot as plt

# this function is used to integrate the dynamics
from scipy.integrate import odeint
# in this class we implement the extended Kalman filter
from ExtendedKalmanFilter import ExtendedKalmanFilter

# discretization time step
# it is used for both integration of the pendulum differential equation and for
# forward Euler method discretization
deltaTime=0.01
# initial condition for generating the simulation data 
x0=np.array([np.pi/3,0.2])

# time steps for simulation
simulationSteps=400
# total simulation time 
totalSimulationTimeVector=np.arange(0,simulationSteps*deltaTime,deltaTime)

# this state-space model defines the continuous dynamics of the pendulum
# this function is passed as an argument of the odeint() function for integrating (solving) the dynamics 
def stateSpaceModel(x,t):
    g=9.81
    # 
    l=1
    dxdt=np.array([x[1],-(g/l)*np.sin(x[0])])
    return dxdt

# here we integrate the dynamics
# the output: "solutionOde" contains time series of the angle and angular velocity 
# these time series represent the time series of the true state that we want to estimate
solutionOde=odeint(stateSpaceModel,x0,totalSimulationTimeVector)
 
# uncomment this if you want to plot the simulation results 
#plt.plot(totalSimulationTimeVector, solutionOde[:, 0], 'b', label='x1')
#plt.plot(totalSimulationTimeVector, solutionOde[:, 1], 'g', label='x2')
#plt.legend(loc='best')
#plt.xlabel('time')
#plt.ylabel('x1(t), x2(t)')
#plt.grid()
#plt.savefig('simulation.png',dpi=600)
#plt.show()

# here we compare the forward Euler discretization with the odeint()
# this is important for evaluating the accuracy of the forward Euler method 

forwardEulerState=np.zeros(shape=(simulationSteps,2))
# set the initial states to match the initial state used in the odeint()
forwardEulerState[0,0]=x0[0]
forwardEulerState[0,1]=x0[1]

# propagate the forward Euler dynamics
for timeIndex in range(simulationSteps-1):
    forwardEulerState[timeIndex+1,:]=forwardEulerState[timeIndex,:]+deltaTime*stateSpaceModel(forwardEulerState[timeIndex,:],timeIndex*deltaTime)
    
# plot the comparison results
plt.plot(totalSimulationTimeVector, solutionOde[:, 0], 'r', linewidth=3, label='Angle - ODEINT')
plt.plot(totalSimulationTimeVector, forwardEulerState[:, 0], 'b', linewidth=2, label='Angle- Forward Euler')
plt.legend(loc='best')
plt.xlabel('time [s]')
plt.ylabel('Angle-x1(t)')
plt.grid()
plt.savefig('comparison.png',dpi=600)
plt.show()


#create the Kalman filter object 
# this is an initial guess of the state estimate
x0guess=np.zeros(shape=(2,1))
x0guess[0]=x0[0]+4*np.random.randn()
x0guess[1]=x0[1]+4*np.random.randn()

# initial value of the covariance matrix
P0=10*np.eye(2,2)
# discretization stpe
dT=deltaTime
# process noise covariance matrix
# note that we do not have the process noise
Q=0.0001*np.eye(2,2)
# measurement noise covariance matrix
# note that we do not have measurement noise in this simulation 
# see driverCodeNoise.py for the performance when the measurement noise
# is affecting the outputs
R=np.array([[0.0001]])

# create the extended Kalman filter object
KalmanFilterObject=ExtendedKalmanFilter(x0guess,P0,Q,R,dT)


# simulate the extended Kalman filter 
for j in range(simulationSteps-1):
    # TWO STEPS
    # (1) propagate a posteriori estimate and covariance matrix
    KalmanFilterObject.propagateDynamics()
    
    # (2) take into account the current measurement and 
    # compute the a posteriori estimate and covarance matrix
    # note that we use the exact solution of the differential 
    # equations as measurements
    # note also that we only measure the angle of the pendulum
    KalmanFilterObject.computeAposterioriEstimate(solutionOde[j, 0])

# Estimates
#KalmanFilterObject.estimates_aposteriori
# Covariance matrices
#KalmanFilterObject.estimationErrorCovarianceMatricesAposteriori
# Kalman gain matrices
#KalmanFilterObject.gainMatrices.append(Kk)
# errors
#KalmanFilterObject.errors.append(error_k)

# extract the state estimates in order to plot the results
estimateAngle=[]
estimateAngularVelocity=[]
for j in np.arange(np.size(totalSimulationTimeVector)):
    estimateAngle.append(KalmanFilterObject.estimates_aposteriori[j][0,0])
    estimateAngularVelocity.append(KalmanFilterObject.estimates_aposteriori[j][1,0])
   
    
# create vectors corresponding to the true values in order to plot the results
trueAngle=solutionOde[:,0]
trueAngularVelocity=solutionOde[:,1]


# plot the results
steps=np.arange(np.size(totalSimulationTimeVector))
fig, ax = plt.subplots(2,1,figsize=(10,15))
ax[0].plot(steps,trueAngle,color='red',linestyle='-',linewidth=6,label='True angle')
ax[0].plot(steps,estimateAngle,color='blue',linestyle='-',linewidth=3,label='Estimate of angle')
ax[0].set_xlabel("Discrete-time steps k",fontsize=14)
ax[0].set_ylabel("Angle",fontsize=14)
ax[0].tick_params(axis='both',labelsize=12)
ax[0].grid()
ax[0].legend(fontsize=14)

ax[1].plot(steps,trueAngularVelocity,color='red',linestyle='-',linewidth=6,label='True angular velocity')
ax[1].plot(steps,estimateAngularVelocity,color='blue',linestyle='-',linewidth=3,label='Angular velocity estimate')
ax[1].set_xlabel("Discrete-time steps k",fontsize=14)
ax[1].set_ylabel("Angular Velocity",fontsize=14)
ax[1].tick_params(axis='both',labelsize=12)
ax[1].grid()
ax[1].legend(fontsize=14)
fig.savefig('estimationResults.png',dpi=600)

