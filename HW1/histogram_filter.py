import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
        corresponding observations. True belief is also given to compare your filters results with actual results. 
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3
    
        ### Your Algorithm goes Below.'''
        
        cmap = np.rot90(cmap,-1)
        belief = np.rot90(belief,-1)
        N = np.array(cmap).shape[0]
        M = np.array(cmap).shape[1]
        out = np.zeros((N,M))
     
        for i in range(N):
            for j in range(M):
                #cases of not out of boundaries!!
                if i+action[0]>=0 and i+action[0]<N and j+action[1]>=0 and j+action[1]<M: # good case
                    out[i+action[0]][j+action[1]] += belief[i][j]*0.9
                    out[i][j] += belief[i][j]*0.1
                else:
                    out[i][j] += belief[i][j]*1   # states remain unchanged
        # perceptual data
        theta = 0
        for i in range(N):
            for j in range(M):
                if cmap[i][j] == observation:  # correct data
                    out[i][j] = 0.9*out[i][j]
                    theta += out[i][j]
                else:                              # incorrect data
                    out[i][j] = 0.1*out[i][j]
                    theta += out[i][j]
        out /= theta
        tmp = 0
        for i in range(N):
            for j in range(M):
                if out[i][j] > tmp:
                    tmp = out[i][j]
                    state = [i,j]
        return np.rot90(out, k=1), np.array(state)