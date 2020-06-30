import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from math import pi,exp
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from math import exp
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import norm
import random
from collections import Counter
#TODO
class glvq():

    def __init__(self,max_prototypes_per_class=5,learning_rate=2,strech_factor=1,placement_strategy=None, move_only_winner=False, set_winner=False, inhibit_wminus=False, inhibit_wminus_params={'std_dev': 1, 'min_update': 0.05, 'inhibition_step': 1}, buffered=False, buffer_size=50,buffer_proba=0.5, weighted_rand_winner_selection=False, weighted_rand_looser_selection=False, knn=None):
        """Constructor takes some additional arguments like max_prototypes_per_class, learning_rate. Additionally a
        placement strategy can passed as a callback. The function takes one sample x and corresponding label y and
        returns True if a new prototype should be placed otherwise false. Then the prototype is moved (see fit sample)."""
        self.max_prototypes_per_class = sys.maxsize if max_prototypes_per_class is None else max_prototypes_per_class
        self.learning_rate = learning_rate
        self.strech_factor = strech_factor
        self.placement_strategy = placement_strategy if placement_strategy is not None else self.placement_always
        self.move_only_winner = move_only_winner
        self.set_winner = set_winner
        self.inhibit_wminus = inhibit_wminus
        self.inhibition_std_dev = inhibit_wminus_params['std_dev']
        self.inhibition_min_update = inhibit_wminus_params['min_update']
        self.inhibition_step = inhibit_wminus_params['inhibition_step']
        self.buffered = buffered
        self.buffer_size = buffer_size
        self.buffer_proba = buffer_proba
        self.weighted_rand_winner_selection = weighted_rand_winner_selection
        self.weighted_rand_looser_selection = weighted_rand_looser_selection
        self.knn = knn


    @staticmethod
    def inhibition_function(x, std_dev, min_update):
        res_fun = norm.pdf(x, 0, std_dev)
        res_fun /= norm.pdf(0, 0, std_dev)
        res_fun *= 1 - min_update
        res_fun += min_update
        return res_fun

    def partial_fit(self,x,y):
        '''just a synomyme that follows scikit learn naming conventions'''
        return self.fit(x,y)

    def fit(self,x,y):
        """fit samples x and y incrementally. x and y are saved together with the yet trained samples in self.x and self.y"""
        if len(x) != len(y):
            raise Exception('feature and label vectors have to have the same size')
        if len(x) == 0:
            raise Exception('can not train empty sequence')
        x = np.array(x)
        y = np.array(y)
        feat_dim = x.shape[1]
        if not hasattr(self,'x'):
            self.x = x
            self.y = y
            self.prototypes = np.zeros(shape=(0,feat_dim))
            self.labels = np.array([])
            if self.inhibit_wminus:
                self.inhibition_states = np.array([])
        else:
            self.x = np.vstack((self.x,x))
            self.y = np.hstack((self.y,y))

        if self.max_prototypes_per_class == sys.maxsize: # classifier is a knn, hack for speed improvements
            self.prototypes = np.vstack((self.prototypes, x))
            self.labels = np.hstack((self.labels, y))
        else:
            for xi,yi in zip(x,y):
                if self.buffered:
                    if not hasattr(self, 'buffer_x'):
                        self.buffer_x = np.zeros((0,feat_dim))
                        self.buffer_y = np.array([])
                    if random.random() < self.buffer_proba: # buffer it
                        self.buffer_x = np.vstack((self.buffer_x, [xi]))
                        self.buffer_y = np.hstack((self.buffer_y, [yi]))
                        if len(self.buffer_x) >= self.buffer_size: # train random sample from buffer
                            i = random.randint(0,len(self.buffer_x)-1)
                            self.fit_sample(self.buffer_x[i], self.buffer_y[i])
                            self.buffer_x = np.delete(self.buffer_x, i, axis=0)
                            self.buffer_y = np.delete(self.buffer_y, i, axis=0)
                    else: # train sample without buffering it
                        self.fit_sample(xi, yi)
                else: # unbuffered (this is the regular case)
                    self.fit_sample(xi,yi)


    def placement_adaptive(self,xi,yi):
        """if the new sample got classified correctly before training it, no prototype is inserted."""
        return self.predict([xi]) != [yi]

    def placement_certainty_adaptive(self,xi,yi,thresh=0.8):
        """if the new sample got classified correctly before training it, no prototype is inserted."""
        return self.predict([xi]) != [yi] or self.predict_proba([xi]) < thresh

    def placement_always(self,xi,yi):
        """always insert new prototype"""
        return True

    def fit_sample(self,xi,yi):
        """fit a specific sample incrementally. Checks if a new prototype is neccessary (with self.placement_strategy).
        If so, a new prototype is inserted, otherwise the nearest prototype is moved corresponding to GLVQ update rule.
        """
        # print("fit sample of class "+str(yi))
        num_prototypes_per_class = 0 if len(self.labels) == 0 else len(np.where(self.labels == yi)[0])
        if (num_prototypes_per_class == 0 or self.placement_strategy(xi,yi)) \
                and not num_prototypes_per_class >= self.max_prototypes_per_class:  # add new
            # print('PLACE NEW prototype for class '+str(yi)+' (number of prototypes for this class: '+str(num_prototypes_per_class)+')')
            self.prototypes = np.vstack((self.prototypes, xi))
            self.labels = np.hstack((self.labels, yi))
            if self.inhibit_wminus:
                self.inhibition_states = np.hstack((self.inhibition_states, 0))
            #print("adding prototype for class" + str(yi))
        elif len(set(self.labels)) > 1:  # move prototype
            # print('MOVE EXISTING prototype for class '+str(yi)+' (number of prototypes for this class: '+str(num_prototypes_per_class)+')')
            proto_dist = self.dist(np.array([xi]), self.prototypes)
            proto_dist = proto_dist[0]

            # find out nearest proto of same class and different class
            smallest_dist_wrong = float("inf")
            smallest_dist_right = float("inf")
            w1i = -1
            w2i = -1
            for i, p in enumerate(proto_dist):
                if self.labels[i] == yi and smallest_dist_right > p:
                    smallest_dist_right = p
                    w1i = i
                if self.labels[i] != yi and smallest_dist_wrong > p:
                    smallest_dist_wrong = p
                    w2i = i

            if self.weighted_rand_winner_selection:
                winner_mask = self.labels == yi
                winner_dists = proto_dist[winner_mask]
                winner_is = np.where(winner_mask)[0]
                from common.math_helper import weighted_choice
                try:
                    w1i = weighted_choice(winner_is, 1, winner_dists, True, True)[0]
                except ValueError:
                    print('idk')
                ## small test for weighted selection
                #wc = weighted_choice(winner_is, 10000, winner_dists, True, True)
                #asdf = [list(wc).count(x) for x in winner_is]
            if self.weighted_rand_looser_selection:
                looser_mask = self.labels != yi
                looser_dists = proto_dist[looser_mask]
                looser_is = np.where(looser_mask)[0]
                from common.math_helper import weighted_choice
                w2i = weighted_choice(looser_is, 1, looser_dists, True, True)[0]



            w1 = self.prototypes[w1i]
            w2 = self.prototypes[w2i]
            d1 = proto_dist[w1i]
            d2 = proto_dist[w2i]

            mu = (d1 - d2) / (d1 + d2)

            factor_winner = (d2 / ((d1 + d2) * (d1 + d2)))
            factor_looser = (d1 / ((d1 + d2) * (d1 + d2)))

            # sigm = (1/(1+exp(-mu)))
            derive = exp(-mu * self.strech_factor) / (
            (exp(-mu * self.strech_factor) + 1) * (exp(-mu * self.strech_factor) + 1))
            #print('mu: ' + str(mu) + ' derive: ' + str(derive))
            # GLVQ
            if self.set_winner:
                self.prototypes[w1i] = xi
            else:
                self.prototypes[w1i] = w1 + self.learning_rate * derive * factor_winner * (xi - w1)
                if self.inhibit_wminus:
                    self.inhibition_states[w1i] = max(0, self.inhibition_states[w1i] - self.inhibition_step)
                if not self.move_only_winner:
                    multiplier = 1
                    if self.inhibit_wminus:
                        multiplier = self.inhibition_function(self.inhibition_states[w2i], self.inhibition_std_dev, self.inhibition_min_update)
                        if self.inhibition_function(self.inhibition_states[w2i], self.inhibition_std_dev, self.inhibition_min_update) != self.inhibition_min_update:
                            self.inhibition_states[w2i] += self.inhibition_step
                        #print('wminus multipliers: %f (all: %s)' %(multiplier, self.inhibition_states))
                    self.prototypes[w2i] = w2 - self.learning_rate * derive * factor_looser * (xi - w2) * multiplier
            #print('derive ' + str(derive))
            #print('move p1 from ' + str(w1) + ' to ' + str(self.prototypes[w1i]))
            #print('move p2 from ' + str(w2) + ' to ' + str(self.prototypes[w2i]))

        else:
            pass
            # print('cant move because only one labeled class: '+str(set(self.labels)))

    def dist(self,x,y):
        """calculates the distance matrix used for determine the winner and looser prototype"""
        return cdist(x,y,'euclidean')


    def predict_proba_full_matrix(self,x):
        return self.predict_proba(x,full_matrix=True)

    def predict_proba(self,x,full_matrix=False,return_winning_prototype_i=False,return_relsims=False):
        """returns the relative distance of prototypes to samples from x"""


        if not hasattr(self,'labels') or len(set(self.labels)) < 2:
            rtn = np.array([0] * len(x)) if not full_matrix else np.zeros((len(x),1))
            if return_winning_prototype_i:
                return rtn, np.array([None] * len(x))
            else:
                return rtn

        num_classes = len(set(self.labels))

        ds = self.dist(x,self.prototypes)
        relsims = []
        winning_prototype_is = []
        for d in ds:
            if full_matrix:
                protos = self.get_win_loose_prototypes(d,num_classes)
                # proto_relsims = np.zeros((num_classes))
                from common.math_helper import get_proba_from_dists
                proto_relsims = get_proba_from_dists([d[i] for i in protos])
                ## old faulty code
                # # step through the loosers and calculate relsim for all to get full certainty matrix
                # for i,p in enumerate(protos[1:]):
                #     proto_relsims[protos[i]] = (d[p] - d[protos[0]]) / (d[p] + d[protos[0]])
                #     if not return_relsims:
                #         proto_relsims[protos[i]] = proto_relsims[protos[i]] * 0.5 + 0.5
                relsims.append(proto_relsims)
                winning_prototype_is.append(protos[0])
            else:
                winner,looser = self.get_win_loose_prototypes(d)
                if return_relsims:
                    relsims.append((d[looser]-d[winner])/(d[looser]+d[winner]))
                else:
                    relsims.append(((d[looser]-d[winner])/(d[looser]+d[winner])) * 0.5 + 0.5)
                winning_prototype_is.append(winner)

        if return_winning_prototype_i:
            return (np.array(relsims),np.array(winning_prototype_is))
        else:
            return np.array(relsims)

    def get_win_loose_prototypes(self,dists,n=2):
        """get the winning prototype and the n-1 loosing prototypes"""
        ds = np.argsort(dists)
        # the classes already included into prototype list
        labels_included = []
        prototypes_i = []

        for id, d in enumerate(ds):
            if not self.labels[d] in labels_included:
                labels_included.append(self.labels[d])
                prototypes_i.append(d)
                if len(prototypes_i) >= n:
                    break
        return prototypes_i

    def predict(self,x):
        """predicts samples from x"""
        if not hasattr(self,'labels') or len(set(self.labels)) < 2:
            return [-1] * len(x)
        if self.knn is None: # consider only nearest prototype
            return np.array(self.labels[np.argmin(self.dist(x,self.prototypes), axis=1)])
        else: # consider k nearest neighbor prototypes for prediction
            nearest_i = np.argsort(self.dist(x, self.prototypes), axis=1)[:,:self.knn]
            return np.array([Counter(self.labels[xx]).most_common(1)[0][0] for xx in nearest_i])

    def predict_sample(self,x):
        """predicts a single sample"""
        return self.predict(x[np.newaxis])

    def score(self,x,y):
        """0/1 loss"""
        y_pred = self.predict(x)
        return float(len(np.where(y==y_pred)[0]))/len(x)


    def visualize_2d(self,ax=None,dir=None):
        """draws a nice little visualization about the actual train state with matplotlib"""
        if not hasattr(self,'pltCount'):
            self.pltCount = 0
        if ax is None:
            #f = plt.figure()
            ax = plt.gca()

        if dir is None:
            dir = '.'

        ax.cla()

        plt.ion()
        some_colors = ['red','green','blue','yellow','orange','pink','black','brown']
        unique_labels = np.unique(self.labels)
        
        
        pred = self.predict(self.x)
        for x,y in zip(self.x,pred):
            ax.scatter(x[0],x[1],c='grey')#some_colors[int(y)]
        for p,l in zip(self.prototypes,self.labels):
            ax.scatter(p[0],p[1],c=some_colors[np.where(unique_labels == l)[0][0]],marker='D',s=80,edgecolors='black')
        plt.pause(0.001)
        plt.savefig(os.path.join(dir,'plt'+str(self.pltCount)+'.png'),format='png')
        self.pltCount +=1

        plt.ioff()

if __name__ == '__main__':
    """a small test program wich demonstrate the use of the classifier"""
    # x,y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
    #                                      n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=0.5,
    #                                      hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=43)
    xt = np.random.multivariate_normal((0,0), [[1,0],[0,1]], 500)
    xt = np.vstack((xt,np.random.multivariate_normal((2,0), [[1,0],[0,1]], 500)))
    yt = np.array(['class1' for _ in range(500)])
    yt = np.hstack((yt,['class2' for _ in range(500)]))

    x_train, x_test, y_train, y_test = train_test_split(xt,yt)

    plt.scatter(xt[:500,0],xt[:500,1],c='blue')
    plt.scatter(xt[500:,0],xt[500:,1],c='red')
    plt.show()

    b = glvq(max_prototypes_per_class=5,learning_rate=5)

    for i,(x,y) in enumerate(zip(x_train,y_train)):
        b.fit([x],[y])
        #b.visualize_2d()


    # # train whole data set at once
    # b.fit(x_train, y_train)

    print(b.predict(x_test))
    print(y_test)
    print(b.predict_proba_full_matrix(x_test))
    print('score: '+str(b.score(x_test,y_test)))
