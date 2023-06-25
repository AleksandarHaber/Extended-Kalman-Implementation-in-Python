# -*- coding: utf-8 -*-
"""
This is an implementation of the extended Kalman filter equations 
that are derived and explained here

The class defined in this file is used in the driver code file: driverCode.py

The webpage tutorials accompanying the developed code are given here:
    
Part 1:
https://aleksandarhaber.com/extended-kalman-filter-tutorial-with-disciplined-python-codes/

Part 2:
https://aleksandarhaber.com/extended-kalman-filter-tutorial-with-example-and-disciplined-python-codes-part-ii-python-codes/

Author: Aleksandar Haber 
Last Revision: June, 2023
"""

import numpy as np 

class ExtendedKalmanFilter(object):
    
    # x0 - initial guess of the state vector - this is the initial a posteriori estimate
    # P0 - initial guess of the covariance matrix of the state estimation error
    # Q  - covariance matrix of the process noise 
    # R  - covariance matrix of the measurement noise
    # dT - discretization period for the forward Euler method
    
    def __init__(self,x0,P0,Q,R,dT):
        
        # initialize vectors and matrices
        self.x0=x0
        self.P0=P0
        self.Q=Q
        self.R=R
        self.dT=dT
        
        # model parameters
        # gravitational constant
        self.g=9.81
        # length of the pendulum 
        self.l=1
        
        # this variable is used to track the current time step k of the estimator 
        # after every measurement arrives, this variables is incremented for +1 
        self.currentTimeStep=0
        
        # this list is used to store the a posteriori estimates hat{x}_k^{+} starting from the initial estimate 
        # note: the list starts from hat{x}_0^{+}=x0 - where x0 is an initial guess of the estimate provided by the user
        self.estimates_aposteriori=[]
        self.estimates_aposteriori.append(x0)
        
        # this list is used to store the a apriori estimates hat{x}_k^{-} starting from hat{x}_1^{-}
        # note: hat{x}_0^{-} does not exist, that is, the list starts from the time index 1
        self.estimates_apriori=[]
        
        # this list is used to store the a posteriori estimation error covariance matrices P_k^{+}
        # note: the list starts from P_0^{+}=P0, where P0 is the initial guess of the covariance provided by the user
        self.estimationErrorCovarianceMatricesAposteriori=[]
        self.estimationErrorCovarianceMatricesAposteriori.append(P0)
        
        # this list is used to store the a priori estimation error covariance matrices P_k^{-}
        # note: the list starts from P_1^{-}, that is, it starts from the time index 1
        self.estimationErrorCovarianceMatricesApriori=[]
        
        # this list is used to store the Kalman gain matrices K_k
        self.gainMatrices=[]
         
        # this list is used to store prediction errors error_k=y_k-self.outputEquation(x_k^{-})
        self.errors=[]
    
    
    # here is the continuous state-space model
    # inputs:
    #       x - state vector 
    #       t - time
    # NOTE THAT WE ARE NOT USING time since the dynamics is time invariant
    # output: 
    #       dxdt - the value of the state function (derivative of x)
    def stateSpaceContinuous(self,x,t):
        dxdt=np.array([[x[1,0]],[-(self.g/self.l)*np.sin(x[0,0])]])
        return dxdt
    
    # this function defines the discretized state-space model 
    # we use the forward Euler discretization 
    # input: 
    #       x_k   - current state x_{k}
    # output:
    #       x_kp1 - state propagated in time x_{k+1}
    def discreteTimeDynamics(self,x_k):
        # note over here that we are not using "self.currentTimeStep*self.DT" since the dynamics is time invariant
        # however, you might need to use this argument if your dynamics is time varying
        x_kp1=x_k+self.dT*self.stateSpaceContinuous(x_k,self.currentTimeStep*self.dT)
        return x_kp1
    
    # this function returns the Jacobian of the discrete-time state equation 
    # evaluated at x_k
    # That is, it returns the matrix A
    # input: 
    #       x_k - state 
    # output: 
    #       A - the Jacobian matrix of the state equation with respect to state
    def jacobianStateEquation(self,x_k):
        A=np.zeros(shape=(2,2))
        A[0,0]=1
        A[0,1]=self.dT
        A[1,0]=-self.dT*(self.g/self.l)*np.cos(x_k[0,0])
        A[1,1]=1
        return A
    
    # this function returns the Jacobian of the output equation 
    # evaluated at x_k
    # That is, it returns the matrix C
    # Note that since in the case of the pendulum the output is a linear function 
    # and consequently, we actually do not use x_k
    # however, in the case of nonlinear output functions we need x_k
    # input: 
    #      x_k - state 
    # output: 
    #      C   - the Jacobian matrix of the output equation with respect to state
    def jacobianOutputEquation(self,x_k):
        C=np.zeros(shape=(1,2))
        C[0,0]=1
        return C
    
    # this is the output equation
    # input: 
    #       x_k - state
    # output: 
    #       x_k[0]- output value at the current state
    def outputEquation(self,x_k):
        return x_k[0]
    
    # this function propagates x_{k-1}^{+} through the model to compute x_{k}^{-}
    # this function also propagates P_{k-1}^{+} through the time-covariance model to compute P_{k}^{-}
    # at the end, this function increments the time index self.currentTimeStep for +1
     
    def propagateDynamics(self):
        # propagate the a posteriori estimate to compute the a priori estimate
        xk_minus=self.discreteTimeDynamics(self.estimates_aposteriori[self.currentTimeStep])
        # linearize the dynamics at the a posteriori estimate 
        Akm1=self.jacobianStateEquation(self.estimates_aposteriori[self.currentTimeStep])
        # propagate the a posteriori covariance matrix in time to compute the a priori covariance
        Pk_minus=np.matmul(np.matmul(Akm1,self.estimationErrorCovarianceMatricesAposteriori[self.currentTimeStep]),Akm1.T)+self.Q
        
        # memorize the computed values and increment the time step
        self.estimates_apriori.append(xk_minus)
        self.estimationErrorCovarianceMatricesApriori.append(Pk_minus)
        self.currentTimeStep=self.currentTimeStep+1
    
    # this function computes the a posteriori estimate by using the measurements
    # this function should be called after propagateDynamics() because the time step should be increased and states and covariances should be propagated         
    # input:
    #       currentMeasurement - measurement at the time step k
    def computeAposterioriEstimate(self,currentMeasurement):
                
        
        # linearize the output equation at the a priori estimate for the time step k
        Ck=self.jacobianOutputEquation(self.estimates_apriori[self.currentTimeStep-1]) 
        
        # compute the Kalman gain matrix
        # keep in mind that the a priori indices start from k=1, that is why we index a priori quantities with "self.currentTimeStep-1"
        Smatrix= self.R+np.matmul(np.matmul(Ck,self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep-1]),Ck.T)
        # Kalman gain matrix
        Kk=np.matmul(self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep-1],np.matmul(Ck.T,np.linalg.inv(Smatrix)))
        
        # update the estimate
        # prediction error
        error_k=currentMeasurement-self.outputEquation(self.estimates_apriori[self.currentTimeStep-1])
        # a posteriori estimate
        xk_plus=self.estimates_apriori[self.currentTimeStep-1]+np.matmul(Kk,np.array([error_k]))
        
        # update the covariance matrix
        # a posteriori covariance matrix update 
        IminusKkC=np.eye(self.x0.shape[0])-np.matmul(Kk,Ck)
        Pk_plus=np.matmul(IminusKkC,np.matmul(self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep-1],IminusKkC.T))+np.matmul(Kk,np.matmul(self.R,Kk.T))
        
        # update the lists that store the vectors and matrices
        # Kalman gain matrix
        self.gainMatrices.append(Kk)
        # errors
        self.errors.append(error_k)
        # a posteriori estimates
        self.estimates_aposteriori.append(xk_plus)
        # a posteriori covariance matrix
        self.estimationErrorCovarianceMatricesAposteriori.append(Pk_plus)
