# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util, math
from game import Directions, Actions, Grid
import game
import os, subprocess
import re
from pacman import GameState
from typing import Optional
from sys import platform
import collections
from util import manhattanDistance
import game
import cProfile
from functools import lru_cache

###################
# Path Definition #
###################

CD = os.path.dirname(os.path.abspath(__file__))

# On server
if platform.startswith("linux"):
    FF_PATH = f"{CD}/ff"

# In local environment
elif platform.startswith("darwin"):
    FF_PATH = f"{CD}/my_ff/ff"

else:
    print("Error in identifying OS")
    FF_PATH = f"{CD}/ff"

OFFENSIVE_DOMAIN_FILE_PATH = f"{CD}/offensive-domain.pddl"
DEFENSIVE_DOMAIN_FILE_PATH = f"{CD}/defensive-domain.pddl"

#################
# Team creation #
#################


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="MCTSAgentOffensive",
    second="MCTSAgentOffensive",
):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    path = "agents/ff"
    os.chmod(path, os.stat(path).st_mode | 0o111)
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

## MCTS Agents


class MCTSAgentOffensive(CaptureAgent):
    """
    An agent that uses Monte-Carlo Tree Search (MCTS) to choose actions.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        """
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        """
        CaptureAgent.registerInitialState(self, gameState)

        """
        Your initialization code goes here, if you need any.
        """

        self.MAX_CARRY = (
            3  # maximum pellets that an agent will collect before returning home
        )
        self.carrying = 0  # number of pellets the agent is currently carrying
        self.boundary = self.getBoundary(gameState)  # get boundary of home side
        self.startingFoodCount = len(
            self.getFood(gameState).asList()
        )  # number of total opponent pellets at beginning of game
        self.recentlyVisited = collections.deque(maxlen=5)  # recently visited locations

        self.team = self.getTeam(gameState)
        self.opponents = self.getOpponents(gameState)
        self.repetitionPenalty = False

        self.prevNode = None
        self.investigating = False
        self.foodEatenPos = None
        self.closestOpponentFoodPos = None
        self.minOpponentDistance = None
        self.prevMinOpponentDistance = None
        self.prevNearbyOpponents = []

    # taken from baselineTeam.py
    def getBoundary(self, gameState):
        boundary_location = []
        height = gameState.data.layout.height
        width = gameState.data.layout.width
        for i in range(height):
            if self.red:
                j = int(width / 2) - 1
            else:
                j = int(width / 2)
            if not gameState.hasWall(j, i):
                boundary_location.append((j, i))
        return boundary_location

    def chooseAction(self, gameState):
        """
        Chooses an action based on the results of a MCTS
        """

        # find all visible opponents
        self.visibleOpponents = []
        for opponent in self.opponents:
            if gameState.getAgentPosition(opponent):
                self.visibleOpponents.append(opponent)

        # find all nearby visible opponents
        self.nearbyOpponents = []
        agentPos = gameState.getAgentPosition(self.index)
        for opponentIndex in self.visibleOpponents:
            opponentPos = gameState.getAgentPosition(opponentIndex)
            opponentDistance = abs(agentPos[0] - opponentPos[0]) + abs(
                agentPos[1] - opponentPos[1]
            )
            if opponentDistance <= 10:
                self.nearbyOpponents.append(opponentIndex)

        # keep track of runtime stats
        self.root = Node(gameState)
        if (
            self.root.gameState.getAgentPosition(self.index)
            in self.getBoundary(self.root.gameState)
            or (self.root.gameState.data.agentStates[self.index].isPacman)
            or (self.root.gameState.data.agentStates[self.index].scaredTimer > 0)
        ):
            self.stateEvaluator = offensiveEvaluator(self.root, self)
            mcts = MCTS(self, self.root, mode="offensive")
            action = mcts.search()
        else:
            self.stateEvaluator = defensiveEvaluator(self.root, self)
            mcts = MCTS(self, self.root, mode="defensive")
            action = mcts.search()

        # calculate repetition penalty
        self.repetitionPenalty = False
        self.prevNode = self.root
        self.prevNearbyOpponents = self.nearbyOpponents
        # if (
        #     self.root.gameState.getAgentPosition(self.index) in self.recentlyVisited
        #     and not self.nearbyOpponents
        # ):
        #     self.repetitionPenalty = True

        # add root position to recently visited states
        self.recentlyVisited.append(self.root.gameState.getAgentPosition(self.index))

        return action


class MCTSAgentDefensive(CaptureAgent):
    """
    An agent that uses Monte-Carlo Tree Search (MCTS) to choose actions.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        """
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    """
        CaptureAgent.registerInitialState(self, gameState)

        """
        Your initialization code goes here, if you need any.
        """

        self.MAX_CARRY = (
            3  # maximum pellets that an agent will collect before returning home
        )
        self.carrying = 0  # number of pellets the agent is currently carrying
        self.boundary = self.getBoundary(gameState)  # get boundary of home side
        self.startingFoodCount = len(
            self.getFood(gameState).asList()
        )  # number of total opponent pellets at beginning of game
        self.recentlyVisited = collections.deque(maxlen=5)  # recently visited locations

        self.team = self.getTeam(gameState)
        self.opponents = self.getOpponents(gameState)

        self.prevNode = None
        self.investigating = False
        self.foodEatenPos = None

    # taken from baselineTeam.py
    def getBoundary(self, gameState):
        boundary_location = []
        height = gameState.data.layout.height
        width = gameState.data.layout.width
        for i in range(height):
            if self.red:
                j = int(width / 2) - 1
            else:
                j = int(width / 2)
            if not gameState.hasWall(j, i):
                boundary_location.append((j, i))
        return boundary_location

    def chooseAction(self, gameState):
        """
        Chooses an action based on the results of a MCTS
        """

        # find all visible opponents
        self.visibleOpponents = []
        for opponent in self.opponents:
            if gameState.getAgentPosition(opponent):
                self.visibleOpponents.append(opponent)

        # find all nearby visible opponents
        self.nearbyOpponents = []
        agentPos = gameState.getAgentPosition(self.index)
        for opponentIndex in self.visibleOpponents:
            opponentPos = gameState.getAgentPosition(opponentIndex)
            opponentDistance = abs(agentPos[0] - opponentPos[0]) + abs(
                agentPos[1] - opponentPos[1]
            )
            if opponentDistance <= 10:
                self.nearbyOpponents.append(opponentIndex)

        # keep track of runtime stats
        self.root = Node(gameState)
        self.stateEvaluator = defensiveEvaluator(self.root, self)  # set state evaluator
        mcts = MCTS(self, self.root, mode="defensive")
        action = mcts.search()

        self.prevNode = self.root

        return action


class MCTS:
    """
    A class implementing a Monte-Carlo Tree search for a pacman agent
    """

    def __init__(
        self, agent, root, mode="offensive", maxIterations=10000, timeout=0.03
    ):
        self.agent = agent  # agent using the search
        self.root = root  # current gamestate
        self.mode = mode  # mode (offensive/defensive)
        self.iterations = maxIterations  # a fixed number of search iterations
        self.timeout = timeout

        # print(self.root.gameState)

        # a set of states expanded during this turn's search
        self.expandedStates = set()
        self.expandedStates.add(self.getStateSignature(self.root.gameState))

        # total number of agents
        self.numAgents = self.root.gameState.getNumAgents()

        # index of agent with next turn
        self.nextAgentIndex = (self.agent.index + 1) % self.numAgents

        #  number of expanded nodes in current search
        self.expandedCount = 0

        # initialise root value
        self.root.value = self.agent.stateEvaluator.evaluate(root)

        # max depth of search (for evaluation)
        self.maxDepth = 0

    def search(self):
        startTime = time.time()

        for i in range(self.iterations):
            # print(f"ITERATION {i}")
            # print(f"agent {self.agent.index}")
            # print(f"expanded: {self.expandedStates}")
            elapsedTime = time.time() - startTime  # Calculate elapsed time
            # print(f"time {elapsedTime}")
            if elapsedTime > self.timeout:
                # print(f"ITERATION {i}")
                break

            # print(f"dead: {self.deadStates}")

            expandedNode = self.treePolicy(self.root)

            if expandedNode:
                self.backPropagate(expandedNode)

        # Return the action of the best child of root_node
        # print(f"expanded: {self.expandedCount}")
        print(self.maxDepth)
        # print(
        #     f"values of children: {[(child.value / child.visits, child.action) for child in self.root.children]}"
        # )
        minByAction = {}
        cumulativeByAction = {}
        for child in self.root.children:
            if child.action not in cumulativeByAction:
                cumulativeByAction[child.action] = child.value
            else:
                cumulativeByAction[child.action] += child.value

            if (
                child.action not in minByAction
                or child.value < minByAction[child.action].value
            ):
                minByAction[child.action] = child

        # print([child.value for child in minByAction.values()])

        if not minByAction.values():
            return random.choice(self.root.gameState.getLegalActions(self.agent.index))

        # apply repetition penalty

        chosenChild = max(minByAction.values(), key=lambda child: child.value)
        if chosenChild.value < 0:
            action = max(cumulativeByAction, key=cumulativeByAction.get)
        else:
            action = chosenChild.action

        # print([(child.value, child.action) for child in self.root.children])

        # print(f"action {action}")

        dx, dy = game.Actions.directionToVector(action)
        x, y = self.root.gameState.getAgentPosition(self.agent.index)
        new_x, new_y = int(x + dx), int(y + dy)
        if self.agent.getFood(self.root.gameState)[new_x][new_y]:
            self.agent.carrying += 1
        elif (new_x, new_y) in self.agent.boundary:
            self.agent.carrying = 0

        return action

    def treePolicy(self, node):
        """
        Policy defining how nodes are selected and expanded in each iteration.
        """
        if node is None:
            return None

        # expand nodes using some MAB policy until a leaf node is reached
        while not node.isLeaf():
            parentNode = node

            node = self.pickChild(parentNode)
            if node == None:
                return None

            node.visits += 1

            # print(f"picking children of {self.getStateSignature(parentNode.gameState)} to expand")

            if node is None:
                return None

        # expand leaf node
        self.expand(node)

        if node.depth > self.maxDepth:
            self.maxDepth = node.depth
        # print(f"expanded {self.getStateSignature(node.gameState)}")

        # randomly select a child of the newly expanded node
        # chosenNode = self.pickRandomChild(node)
        # print(f"chosenNode: {self.getStateSignature(chosenNode.gameState)}")

        # return chosenNode
        return node

    def expand(self, parentNode):
        """
        Expand a leaf node
        """
        # update expanded count
        self.expandedCount += 1

        # get child states
        parentNode.children = self.getChildren(parentNode)

        # check that node has children
        if parentNode.children == []:
            # if not, mark the node as dead and return
            parentNode.isDead = True
            self.expandedStates.add(self.getStateSignature(parentNode.gameState))
            return

        # calculate value for each child state
        for child in parentNode.children:
            child.value = self.agent.stateEvaluator.evaluate(child)
            # if (
            #     self.mode == "offensive"
            #     and (not self.agent.nearbyOpponents)
            #     and parentNode.depth == 0
            # ):
            #     if (
            #         child.gameState.getAgentPosition(self.agent.index)
            #         in self.agent.recentlyVisited
            #     ):
            #         print("repetition penalty applied")
            #         child.value *= 0.7
            #         child.repetitionPenalty = True

        # group children by actions
        for child in parentNode.children:
            if child.action not in parentNode.childrenByAction:
                parentNode.childrenByAction[child.action] = []
            parentNode.childrenByAction[child.action].append(child)

        # calculate the minimum score for each action among the children
        for action, children in parentNode.childrenByAction.items():
            parentNode.minByAction[action] = min(child.value for child in children)

        # update the parent's value to be the highest minimum among the actions (minimax)
        parentNode.value = max(parentNode.minByAction.values())

        # record state as expanded to avoid re-expansion
        self.expandedStates.add(self.getStateSignature(parentNode.gameState))

    def backPropagate(self, node):
        """
        Backpropagate rewards up search tree.
        """

        # print(f"backprop node: {self.getNodeRepresentation(node)}")

        while node.parent is not None:
            # print(
            #     f"parent's children: {[self.getNodeRepresentation(child) for child in node.parent.children]}"
            # )
            node.parent.updateValue(node)
            node = node.parent

        node.visits += 1

    def pickChild(self, parentNode):
        """
        Pick a child node using UCB to balance exploitation and exploration
        """
        children = parentNode.children
        # print(
        #     f"All children: {[child.gameState.getAgentPosition(self.agent.index) for child in children]}"
        # )

        if not children:
            return None

        # Get valid children (unvisited, not dead)
        validChildren = []
        for child in children:
            if not child.isDead:
                validChildren.append(child)

        # print(
        #     f"validChildren: {[self.getStateSignature(child.gameState) for child in validChildren]}"
        # )

        # if the node has no valid children, mark it as dead and start next iteration
        if not validChildren:
            parentNode.isDead = True
            return None

        # pick best valid child using UCB
        C = 1.0
        bestChild = max(
            validChildren,
            key=lambda child: self.uctValue(child, parentNode.visits, C),
        )

        return bestChild

    def pickRandomChild(self, node):
        """
        Returns a random child node.
        """
        children = node.children
        if children:
            return random.choice(children)

        return None

    def uctValue(self, node, parentVisits, C):
        """
        Calculates the UCT value using the UCB formula
        """
        if node.visits == 0:
            return float("inf")

        exploit = node.value / node.visits
        explore = C * (math.sqrt(math.log(parentVisits) / node.visits))

        return exploit + explore

    def getStateRepresentation(self, gameState):
        """
        Returns a string representation of the state for debugging purposes.
        """
        position = gameState.getAgentPosition(self.agent.index)
        foodCount = len(self.agent.getFood(gameState).asList())

        return f"({position[0]}, {position[1]}), food: {foodCount}"

    def getNodeRepresentation(self, node):
        """
        Returns a string representation of the state for debugging purposes.
        """
        value = node.value
        action = node.action

        return f"({value}, {action})"

    def getStateSignature(self, gameState):
        """
        Returns a signature for a game state, used to check if that state has been visited
        before in the deep search (when no enemies are visible).
        """
        return hash(gameState)

    def getChildren(self, parentNode):
        """
        Returns all child states in current round of turns, including the agent and any visible opponents.
        """

        # print("getChildStates called")

        def findNextOpponent(startIndex, agentIndex):

            index = (startIndex + 1) % self.numAgents
            if index == agentIndex:
                return None
            while index not in self.agent.nearbyOpponents:
                index = (index + 1) % self.numAgents
                if index == agentIndex:
                    return None
            return index

        # generate child nodes based on agent's moves
        childNodes = []
        childStates = []
        agentActions = parentNode.gameState.getLegalActions(self.agent.index)
        for action in agentActions:
            childState = parentNode.gameState.generateSuccessor(
                self.agent.index, action
            )
            if not self.agent.nearbyOpponents:
                childSignature = self.getStateSignature(childState)
                if childSignature in self.expandedStates:
                    continue
                childNode = Node(
                    childState,
                    depth=(parentNode.depth + 1),
                    parent=parentNode,
                    action=action,
                )
                childNodes.append(childNode)
            else:
                childStates.append((childState, action))

        # consider opponent's moves if visible [HOW TO ONLY CONSIDER NEW?]
        if self.agent.nearbyOpponents:
            currIndex = findNextOpponent(self.agent.index, self.agent.index)
            while not currIndex == None:
                nextGameStates = []
                for childState in childStates:
                    opponentActions = childState[0].getLegalActions(currIndex)
                    for opponentAction in opponentActions:
                        nextGameState = childState[0].generateSuccessor(
                            currIndex, opponentAction
                        )
                        if self.mode == "offensive":
                            # only add opponent moves if they move towards the agent
                            agentPos = childState[0].getAgentPosition(self.agent.index)
                            prevOpponentPos = childState[0].getAgentPosition(currIndex)
                            opponentPos = nextGameState.getAgentPosition(currIndex)
                            prevOpponentDistance = self.agent.distancer.getDistance(
                                agentPos, prevOpponentPos
                            )
                            opponentDistance = self.agent.distancer.getDistance(
                                agentPos, opponentPos
                            )
                            if opponentDistance < prevOpponentDistance:
                                nextGameStates.append((nextGameState, childState[1]))
                        else:
                            rootOpponentPos = (
                                self.agent.root.gameState.getAgentPosition(currIndex)
                            )
                            prevOpponentPos = childState[0].getAgentPosition(currIndex)
                            opponentPos = nextGameState.getAgentPosition(currIndex)
                            prevOpponentDistance = self.agent.distancer.getDistance(
                                rootOpponentPos, prevOpponentPos
                            )
                            opponentDistance = self.agent.distancer.getDistance(
                                rootOpponentPos, opponentPos
                            )
                            if opponentDistance >= prevOpponentDistance:
                                nextGameStates.append((nextGameState, childState[1]))

                childStates = nextGameStates
                currIndex = findNextOpponent(currIndex, self.agent.index)

            for childState in childStates:
                childSignature = self.getStateSignature(childState)
                if not (childSignature in self.expandedStates):
                    childNode = Node(
                        childState[0],
                        depth=(parentNode.depth + 1),
                        parent=parentNode,
                        action=childState[1],
                    )
                    childNodes.append(childNode)

        return childNodes


class Node:
    def __init__(self, gameState, depth=0, agentIndex=None, parent=None, action=None):
        self.gameState = gameState
        self.parent = parent
        self.action = action

        # agent whose turn it is in this node (None means self)
        self.agentIndex = agentIndex

        self.children = []
        self.minByAction = {}
        self.childrenByAction = {}
        self.depth = depth
        self.visits = 1
        self.value = None
        self.eatsFood = False
        self.isDead = False
        self.visibleOpponents = []
        self.nearbyOpponents = []
        self.repetitionPenalty = False

    def isLeaf(self):
        return len(self.children) == 0

    def updateValue(self, child):
        if self.children == []:
            return None

        if child.value < self.minByAction[child.action]:
            self.minByAction[child.action] = child.value
        else:
            newMinForAction = min(
                [c.value for c in self.childrenByAction[child.action]]
            )
            self.minByAction[child.action] = newMinForAction

        self.value = max(self.minByAction.values())
        if self.repetitionPenalty:
            self.value *= 0.7


class stateEvaluator:
    def __init__(self, node, agent):
        self.node = node
        self.agent = agent

        self.rootAgentPosition = self.agent.root.gameState.getAgentPosition(
            self.agent.index
        )
        self.rootNumFoodLeft = self.agent.getFood(self.agent.root.gameState).count()
        self.rootClosestBoundaryPos = self.getClosestPos(
            self.agent.root.gameState, self.agent.boundary
        )
        self.rootDistanceToBoundary = self.agent.distancer.getDistance(
            self.rootAgentPosition, self.rootClosestBoundaryPos
        )
        self.rootClosestFoodPos = self.getClosestPos(
            self.agent.root.gameState,
            self.agent.getFood(self.agent.root.gameState).asList(),
        )
        if self.rootClosestFoodPos:
            self.rootDistanceToClosestFood = self.agent.distancer.getDistance(
                self.rootAgentPosition, self.rootClosestFoodPos
            )
        else:
            self.rootDistanceToClosestFood = None

        # defence
        self.nearestOpponentPosition = None
        self.agent.minOpponentDistance = 9999
        for opponentIndex in self.agent.visibleOpponents:
            if self.agent.root.gameState.data.agentStates[opponentIndex].isPacman:
                self.rootOpponentPosition = self.agent.root.gameState.getAgentPosition(
                    opponentIndex
                )
                opponentDistance = self.agent.distancer.getDistance(
                    self.rootAgentPosition, self.rootOpponentPosition
                )
                if opponentDistance < self.agent.minOpponentDistance:
                    self.agent.minOpponentDistance = opponentDistance
                    self.nearestOpponentPosition = self.rootOpponentPosition

        self.isFoodEaten = self.foodEaten(self.agent.prevNode, self.agent.root)

        if (self.agent.investigating or (self.isFoodEaten)) and not (
            self.rootAgentPosition == self.agent.closestOpponentFoodPos
        ):
            self.agent.investigating = True
            if self.isFoodEaten:
                foodEatenPosCheck = self.getFoodEatenPos(
                    self.agent.prevNode, self.agent.root
                )
            else:
                foodEatenPosCheck = None
            if foodEatenPosCheck != None:
                self.agent.foodEatenPos = foodEatenPosCheck

            self.agent.closestOpponentFoodPos = self.getClosestPos(
                self.agent.root.gameState,
                self.agent.getFoodYouAreDefending(self.agent.root.gameState).asList(),
                somePos=self.agent.foodEatenPos,
            )

            self.rootDistanceToOpponentFood = self.agent.distancer.getDistance(
                self.rootAgentPosition, self.agent.closestOpponentFoodPos
            )
        else:
            self.agent.investigating = False
            self.agent.foodEatenPos = None
            self.agent.closestOpponentFoodPos = None

        self.agent.prevMinOpponentDistance = self.agent.minOpponentDistance

    def evaluate(self):
        raise NotImplementedError("Subclasses should implement the evaluate method!")

    def getClosestPos(self, gameState, pos_list, somePos=None):
        """
        Finds the closest position to the agent from a given list of positions
        """
        min_length = 9999
        min_pos = None
        if somePos == None:
            my_pos = gameState.getAgentPosition(self.agent.index)
        else:
            my_pos = somePos
        for pos in pos_list:
            temp_length = self.agent.distancer.getDistance(my_pos, pos)
            if temp_length < min_length:
                min_length = temp_length
                min_pos = pos
        return min_pos

    def calculateProximityDiscount(self, gameState, myPos):
        """
        Calculates the discount to be applied based on the agent's proximity to a teammate
        """
        for teammate in self.agent.getTeam(gameState):
            teammatePos = gameState.getAgentPosition(teammate)
            if self.agent.index != teammate:
                distance = self.agent.distancer.getDistance(myPos, teammatePos)
                # print(f"distance: {distance}")
                discount = 1 - (1 / (distance + 3))

        return discount

    def foodEaten(self, prevNode, rootNode):
        """
        Returns boolean to indicate if food was eaten last turn
        """
        if prevNode == None:
            return False

        prevNumFood = self.agent.getFoodYouAreDefending(prevNode.gameState).count()
        numFood = self.agent.getFoodYouAreDefending(rootNode.gameState).count()

        return numFood < prevNumFood

    def getFoodEatenPos(self, prevNode, rootNode):
        """
        Returns the positon of the food eaten since last turn, or None
        if one isn't found.
        """
        prevFoodHalfGrid = self.agent.getFoodYouAreDefending(prevNode.gameState)
        currFoodHalfGrid = self.agent.getFoodYouAreDefending(rootNode.gameState)

        if prevFoodHalfGrid == currFoodHalfGrid:
            return None

        eatenFoodPos = self.xorHalfGrids(prevFoodHalfGrid, currFoodHalfGrid)

        return eatenFoodPos

    def xorHalfGrids(self, halfgrid1, halfgrid2):
        """
        Applies an XOR to two halfgrids. Used here for checking if food has been
        eaten.
        """
        result = Grid(halfgrid1.width, halfgrid1.height, False)

        true_coord = (0, 0)

        for y in range(halfgrid1.height):
            for x in range(halfgrid1.width):
                result[x][y] = halfgrid1[x][y] != halfgrid2[x][y]
                if result[x][y]:
                    true_coord = (x, y)

        return true_coord


class offensiveEvaluator(stateEvaluator):
    def evaluate(self, node):
        # discount to be applied exponentially to calculated rewards with depth
        discount = 0.99

        # find the number of food left at the node to be evaluated, as well as the root node
        numFoodLeft = self.agent.getFood(node.gameState).count()

        # find agent position at node and root
        agentPosition = node.gameState.getAgentPosition(self.agent.index)
        if agentPosition == node.gameState.getInitialAgentPosition(self.agent.index):
            node.isDead = True
            return -50

        # account for there being less than two food left at node
        if numFoodLeft < 2:
            node.isDead = True
            if node.parent != None:
                return (node.parent.value) * (discount**node.depth)
            else:
                return 0.001

        closestBoundaryPos = self.getClosestPos(node.gameState, self.agent.boundary)
        distanceToBoundary = self.agent.distancer.getDistance(
            agentPosition, closestBoundaryPos
        )

        # if all but 2 food has been eaten, or it is worthwhile returning
        # with food, return home
        if self.rootNumFoodLeft <= 2 or (
            self.agent.root.gameState.data.agentStates[self.agent.index].isPacman
            and self.agent.carrying > 2
            and (
                self.rootDistanceToClosestFood == None
                or (self.rootDistanceToBoundary < self.rootDistanceToClosestFood)
            )
        ):
            if not (node.gameState.data.agentStates[self.agent.index].isPacman):
                node.isDead = True
                return 150 * (discount**node.depth)
            return (100 / (1 + distanceToBoundary)) * (discount**node.depth)

        # apply a penalty if the agent is too close to it's teammates
        proximityDiscount = self.calculateProximityDiscount(
            node.gameState, agentPosition
        )

        # if food found, give higher reward
        if self.rootNumFoodLeft > numFoodLeft:
            return (
                100
                * (self.rootNumFoodLeft - numFoodLeft)
                * (discount**node.depth)
                * proximityDiscount
            )

        # # otherwise give reward by distance to closest food
        closestFoodPos = self.getClosestPos(
            node.gameState, self.agent.getFood(node.gameState).asList()
        )
        distanceToClosestFood = self.agent.distancer.getDistance(
            agentPosition, closestFoodPos
        )

        return (10 / distanceToClosestFood) * (discount**node.depth) * proximityDiscount


class defensiveEvaluator(stateEvaluator):
    def evaluate(self, node):
        discount = 0.99

        agentPosition = node.gameState.getAgentPosition(self.agent.index)

        if agentPosition == node.gameState.getInitialAgentPosition(self.agent.index):
            node.isDead = True
            return -50

        # find nearest visible opponent, if there is one
        if self.agent.nearbyOpponents:
            self.agent.investigating = False
            if self.nearestOpponentPosition:
                nodeOpponentDistance = self.agent.distancer.getDistance(
                    agentPosition, self.nearestOpponentPosition
                )
                if nodeOpponentDistance == 0:
                    self.isDead = True
                    return 100

                return (10 / nodeOpponentDistance) * (discount**node.depth)

        if self.agent.investigating:
            if (
                self.agent.closestOpponentFoodPos != None
                and self.rootDistanceToOpponentFood != 0
            ):
                distanceToOpponentFood = self.agent.distancer.getDistance(
                    agentPosition, self.agent.closestOpponentFoodPos
                )
                return (10 / (1 + distanceToOpponentFood)) * (discount**node.depth)

        closestBoundaryPos = self.getClosestPos(node.gameState, self.agent.boundary)
        distanceToBoundary = self.agent.distancer.getDistance(
            agentPosition, closestBoundaryPos
        )
        return (10 / (1 + distanceToBoundary)) * (discount**node.depth)
