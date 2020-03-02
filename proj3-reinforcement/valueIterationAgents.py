# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        k = 0
        lastIter = util.Counter()
        thisIter = util.Counter()
        states = self.mdp.getStates()
        for key in states:
            lastIter[key] = 0.0
        
        while(k < self.iterations):
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                actionResults = []
                for action in actions:
                    nextResult = self.mdp.getTransitionStatesAndProbs(state, action)
                    sum = 0.0
                    for i in range(len(nextResult)):
                        sum += nextResult[i][1]*(self.mdp.getReward(state, action, nextResult[i][0]) + self.discount * lastIter[nextResult[i][0]])
                    actionResults.append(sum)
                if(len(actionResults) != 0):
                    thisIter[state] = max(actionResults)
            lastIter = thisIter.copy()
            k += 1

        self.values = thisIter        



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        nextResult = self.mdp.getTransitionStatesAndProbs(state, action)
        sum = 0
        for i in range(len(nextResult)):
            sum += nextResult[i][1]*(self.mdp.getReward(state, action, nextResult[i][0]) + self.discount * self.values[nextResult[i][0]])
        
        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        if(self.mdp.isTerminal(state)):
            return None
        
        max = -float('inf')
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            Q = self.computeQValueFromValues(state, action)
            if(Q > max):
                max = Q
                maxAction = action
        return maxAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        k = 0
        Iter = util.Counter()
        states = self.mdp.getStates()
        for key in states:
            Iter[key] = 0.0
        
        while(k < self.iterations):
            for state in states:
                if(k < self.iterations):
                    actions = self.mdp.getPossibleActions(state)
                    actionResults = []
                    for action in actions:
                        nextResult = self.mdp.getTransitionStatesAndProbs(state, action)
                        sum = 0.0
                        for i in range(len(nextResult)):
                            sum += nextResult[i][1]*(self.mdp.getReward(state, action, nextResult[i][0]) + self.discount * Iter[nextResult[i][0]])
                        actionResults.append(sum)
                    if(len(actionResults) != 0):
                        Iter[state] = max(actionResults)
                k += 1
        

        self.values = Iter   

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #Compute predecessors of all states.
        #Initialize an empty priority queue.
        priorityQueue = util.PriorityQueue()
        predecessorsDict = {}
        states = self.mdp.getStates()
        for s in states:
            predecessorsDict[s] = set()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    nextStates = [i[0] for i in self.mdp.getTransitionStatesAndProbs(state, action)]
                    for nextState in nextStates:
                        if nextState == s:
                            predecessorsDict[s].add(state)

        """
        For each non-terminal state s, do: 
        (note: to make the autograder work for this question, 
        you must iterate over states in the order returned by self.mdp.getStates())
        
        Find the absolute value of the difference between the current value of s in self.values 
        and the highest Q-value across all possible actions from s (this represents what the value should be);
         call this number diff. Do NOT update self.values[s] in this step.
        
        Push s into the priority queue with priority -diff (note that this is negative). 
        We use a negative because the priority queue is a min heap, but we want to prioritize updating states 
        that have a higher error.
        """

        for s in states:
            if self.mdp.isTerminal(s):
                continue
            actions = self.mdp.getPossibleActions(s)
            maxQ = -float('inf')
            for action in actions:
                Q = self.computeQValueFromValues(s, action)
                if(Q > maxQ):
                    maxQ = Q
            
            diff = abs(maxQ - self.values[s])
            priorityQueue.push(s, -diff)

        """ For iteration in 0, 1, 2, ..., self.iterations - 1, do:
            If the priority queue is empty, then terminate.
            Pop a state s off the priority queue.
            Update the value of s (if it is not a terminal state) in self.values.
            
            For each predecessor p of s, do:
            Find the absolute value of the difference between the current value of p in self.values 
            and the highest Q-value across all possible actions from p (this represents what the value should be); 
            call this number diff. Do NOT update self.values[p] in this step.
            f diff > theta, push p into the priority queue with priority -diff (note that this is negative), 
            as long as it does not already exist in the priority queue with equal or lower priority. 
            As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating 
            states that have a higher error."""

        for iteration in range(self.iterations):
            if priorityQueue.isEmpty():
                return
            
            s = priorityQueue.pop()
            
            if(self.mdp.isTerminal(s)):
                continue
            
            actions = self.mdp.getPossibleActions(s)
            maxQ = -float('inf')
            for action in actions:
                Q = self.computeQValueFromValues(s, action)
                if(Q > maxQ):
                    maxQ = Q

            self.values[s]= maxQ

            for p in predecessorsDict[s]:
                actions = self.mdp.getPossibleActions(p)
                maxQ = -float('inf')
                for action in actions:
                    Q = self.computeQValueFromValues(p, action)
                    if(Q > maxQ):
                        maxQ = Q
                diff = abs(maxQ - self.values[p])
                
                if (diff > self.theta):
                    priorityQueue.update(p, -diff)



