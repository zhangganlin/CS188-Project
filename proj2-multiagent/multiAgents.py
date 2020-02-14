# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        print(legalMoves)
        print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print('------------------------------------------------')
        # print (successorGameState)
        # print(newPos)
        # print(newFood)
        # print(newGhostStates)
        # print(newScaredTimes)
        # print('------------------------------------------------\n')
        "*** YOUR CODE HERE ***"
        if action == "Stop":
            stopScore = -20
        else:
            stopScore = 0
        currPos = currentGameState.getPacmanPosition()
        foodDistanceCurr = [manhattanDistance(currPos,foodPos) for foodPos in newFood.asList()]
        foodDistanceNew = [manhattanDistance(newPos,foodPos) for foodPos in newFood.asList()]
        GhostDistance = [manhattanDistance(newPos,ghostState.getPosition()) for ghostState in newGhostStates]
        minGhostDistance = min(GhostDistance)
        if (len(newFood.asList())==0):
            foodNearScore = 0
        elif (-min(foodDistanceCurr)+min(foodDistanceNew)) != 0 :
            foodNearScore = 10/(min(foodDistanceCurr)+min(foodDistanceNew))
        else:
            foodNearScore = 0

        if minGhostDistance == 0:
            ghostNearScore = -100000000000000000
        elif minGhostDistance>5:
            ghostNearScore =0
        else:
            ghostNearScore = -5/minGhostDistance
        print(foodNearScore)
        return successorGameState.getScore()+ghostNearScore+stopScore + foodNearScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def maxValue(self,gameState,depth):
        v = float('-inf')
        legalMoves = gameState.getLegalActions(0)
        for action in legalMoves:
            nextState = gameState.generateSuccessor(0,action)
            v = max(v,self.value(nextState,depth,1))
        return v

    def minValue(self,gameState,depth,ghostIndex):
        v=float('inf')
        legalMoves = gameState.getLegalActions(ghostIndex)
        nextAgentIndex = (ghostIndex+1)%gameState.getNumAgents()
        nextDepth = depth
        if nextAgentIndex == 0:
            nextDepth = depth +1
        for action in legalMoves:
            nextState = gameState.generateSuccessor(ghostIndex,action)
            v = min(v,self.value(nextState,nextDepth,nextAgentIndex))
        return v

    def value(self,gameState,depth,agentIndex):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if depth== self.depth:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState,depth)
        else:
            return self.minValue(gameState,depth,agentIndex)


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()
        scores = []
        for action in legalMoves:
            newState = gameState.generateSuccessor(0,action)
            v = self.value(newState,depth = 0, agentIndex = 1)
            scores.append(v)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def maxValue(self,gameState,depth,alpha,beta):
        v = ('sth',float('-inf'))
        legalMoves = gameState.getLegalActions(0)
        for action in legalMoves:
            nextState = gameState.generateSuccessor(0,action)
            v_1 = max(v[1],self.value(nextState,depth,1,alpha,beta)[1])
            if v_1 is not v[1]:
                v = (action,v_1)
            if v[1] > beta:
                return v
            alpha = max(alpha,v[1])
        return v

    def minValue(self,gameState,depth,ghostIndex,alpha,beta):
        v=('sth',float('inf'))
        legalMoves = gameState.getLegalActions(ghostIndex)
        nextAgentIndex = (ghostIndex+1)%gameState.getNumAgents()
        nextDepth = depth
        if nextAgentIndex == 0:
            nextDepth = depth +1
        for action in legalMoves:
            nextState = gameState.generateSuccessor(ghostIndex,action)
            v_1 = min(v[1],self.value(nextState,nextDepth,nextAgentIndex,alpha,beta)[1])
            if v_1 is not v[1]:
                v=(action,v_1)
            if v[1] < alpha:
                return v
            beta = min(beta,v[1])
        return v

    def value(self,gameState,depth,agentIndex,alpha,beta):
        if gameState.isWin() or gameState.isLose():
            return ('sth',self.evaluationFunction(gameState))
        if depth== self.depth:
            return ('sth',self.evaluationFunction(gameState))
        if agentIndex == 0:
            return self.maxValue(gameState,depth,alpha,beta)
        else:
            return self.minValue(gameState,depth,agentIndex,alpha,beta)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()
        scores = []
        alpha = float('-inf')
        beta = float('inf')
        val = self.value(gameState,0,0,alpha,beta)

        return val[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maxValue(self,gameState,depth):
        v = float('-inf')
        legalMoves = gameState.getLegalActions(0)
        for action in legalMoves:
            nextState = gameState.generateSuccessor(0,action)
            v = max(v,self.value(nextState,depth,1))
        return v

    def meanValue(self,gameState,depth,ghostIndex):
        v=float('inf')
        legalMoves = gameState.getLegalActions(ghostIndex)
        nextAgentIndex = (ghostIndex+1)%gameState.getNumAgents()
        nextDepth = depth
        if nextAgentIndex == 0:
            nextDepth = depth +1
        total = 0
        for action in legalMoves:
            nextState = gameState.generateSuccessor(ghostIndex,action)
            total += self.value(nextState,nextDepth,nextAgentIndex)
        return total/len(legalMoves)

    def value(self,gameState,depth,agentIndex):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if depth== self.depth:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState,depth)
        else:
            return self.meanValue(gameState,depth,agentIndex)
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()
        scores = []
        for action in legalMoves:
            newState = gameState.generateSuccessor(0,action)
            v = self.value(newState,depth = 0, agentIndex = 1)
            scores.append(v)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    Food = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    currPos = currentGameState.getPacmanPosition()
    capsules = currentGameState.getCapsules()
    foodDistanceCurr = [manhattanDistance(currPos,foodPos) for foodPos in Food.asList()]
    capsuleDistanceCurr = [manhattanDistance(currPos,cap) for cap in capsules]
    GhostDistance = [manhattanDistance(currPos,ghostState.getPosition()) for ghostState in newGhostStates]
    minGhostDistance = min(GhostDistance)
    if (len(Food.asList())==0):
        foodNearScore = 0
    elif min(foodDistanceCurr)!= 0 :
        foodNearScore = 10/min(foodDistanceCurr)
    else:
        foodNearScore = 0

    if (len(capsules)==0):
        capsuleNearScore = 0
    elif min(capsuleDistanceCurr)!= 0 :
        capsuleNearScore = 5/min(capsuleDistanceCurr)
    else:
        capsuleNearScore = 5

    if minGhostDistance == 0:
        ghostNearScore = -100000000000000000
    elif minGhostDistance>5:
        ghostNearScore =0
    else:
        ghostNearScore = -5/minGhostDistance
    return  foodNearScore+ghostNearScore+capsuleNearScore  +currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
