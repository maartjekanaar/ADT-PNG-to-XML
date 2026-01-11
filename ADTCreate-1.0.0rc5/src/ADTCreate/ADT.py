# Developed by Jayme Hebinck, for bachelor thesis Leiden University.
# This file is part of the ADT Create project, found on: https://github.com/MainJay/ADTPythonLibrary/.

from .ADTSatisfiability import _Satisfiability
from .GenerateADTerm import _GenADTerm
from .CompareADTerms import _CompADTerms
from .ADTXML import _XML

from typing import List, Tuple, Union, Optional
from xml.dom import minidom


class ADT:
    """
    Objects of class `ADT` form individual nodes. These nodes are connected to each other using the class variables to
    form an ADT. This class forms the center point of **ADT Create**, and therefore links objects of all classes
    together in this class.
    :param idNumber: Determines the ID number (`int`) of a node. ID numbers enable the introduction of `twin nodes`, meaning the occurances of duplicate, independent nodes.
    :param type: Determines the type of a node (`0` for attack, `1` for defense).
    :param refinement: Determines the refinement of a node (`0` for disjunctive (OR), `1` for conjunctive (AND)).
    :param label: Determines the label of a node.
    :param parent: Determines the parent of a node (`ADT` object, `0` for no parent).
    :param root: Contains two variables in `List`. First element is a `Bool` whether this node is the root node. Second element is an `ADT` object of the root node.
    """
    usedIDs = []

    def __init__(self, idNumber: int, type: int, refinement: int, label: str, parent: Optional['ADT'], root: List[Tuple[bool, Optional['ADT']]]) -> None:
        """
        Constructor of `ADT` class. Initializes values for all membervariables needed.
        :param idNumber: Determines the ID number (`int`) of a node. ID numbers enable the introduction of `twin nodes`, meaning the occurances of duplicate, independent nodes.
        :param type: Determines the type of a node (`0` for attack, `1` for defense).
        :param refinement: Determines the refinement of a node (`0` for disjunctive (OR), `1` for conjunctive (AND)).
        :param label: Determines the label of a node.
        :param parent: Determines the parent of a node (`ADT` object, `0` for no parent).
        :param root: Contains two variables in `List`. First element is a `Bool` whether this node is the root node. Second element is an `ADT` object of the root node.
        """
        # Constructor of class ADT, representing a single ADT node.
        self.children = []
        self.type = type
        self.refinement = refinement
        self.id = idNumber
        self.label = label
        self.parent = parent
        self.root = root
        self.leafNode = False # Check whether this node is a leaf node
        self.satisfiability = False # States if the ADT with this node as root node is satisfiable
        self.ADTerm = "" # ADTerm of ADT witht he current node as root node.
        self.LSEquivalenceThreshold = 0.2 # Levenshtein Distance Equivalence Threshold Value
        self.count = 0 # Counter for how many nodes satisfiability is determined of
        self._hasCounterChild = False  # If a node has one counter child node, private variable

        # Class instances:
        self._XML = _XML(self)
        self._Satisfiability = _Satisfiability(self)
        self._GenADTerm = _GenADTerm(self)
        self._CompADTerms = _CompADTerms(self)

    def addChild(self, typeChild: int, refinementChild: int, labelChild: str, currentChild: 'ADT', tree: 'ADT') -> Tuple[bool, Optional['ADT']]:
        """
        Adds a child node with the given parameters as new node to the `tree`, which is the root of the ADT.
        Child node is given its own unique ID.

        :param typeChild: Type of the child node (`0` for attack, `1` for defense).
        :param refinementChild: Refinement of the child node (`0` for disjunctive (OR), `1` for conjunctive (AND)).
        :param labelChild: Label of the child node.
        :param currentChild: `ADT` object of the parent node, to which this child node is connected.
        :param tree: Root node of the ADT this node is connected to.
        :return:`bool`, `ADT`/`None`: `True` and `ADT` object of new child if child is added successfully, `False` and `None` otherwise.
        """
        idNumber = self.usedIDs[-1] + 1 # Take latest ID and increment by one.
        newChild = ADT(idNumber, int(typeChild), int(refinementChild), labelChild, currentChild, [False, tree])
        # Firstly check if this child can be added in the boundaries of Countermeasure Prevention.
        if not self._checkCounterMeasure(newChild):
            return False, None
        # Set this child as leaf node, and if parent node was leaf node, it is not anymore.
        newChild.leafNode = True
        self.children.append(newChild)
        if self.leafNode is True:
            self.leafNode = False
        self.usedIDs.append(idNumber) # Add latest ID to this list.
        return True, newChild

    def delChild(self, id: int) -> bool:
        """
        Deletes a child node with the given ID.

        :param id: ID of child node.
        :return: `bool`: True if child node is deleted successfully, False otherwise.
        """
        # ID of child first has to be found in IDs of current children.
        found = False
        for child in self.children:
            if child.id == int(id):
                found = True
                childToRemove = child
        if not found:
            return False
        # If found, remove it. Set parent node to leaf node if no other children.
        self.children.remove(childToRemove)
        if int(id) in self.usedIDs:
            self.usedIDs.remove(int(id)) # Remove this ID from usedIDs list.
        if len(self.children) == 0:
            self.leafNode = True
        # If this was a countermeasure node, turn the countermeasure variable off.
        if (self.type == 0 and self._hasCounterChild and child.type == 1) or (self.type == 1 and self._hasCounterChild and child.type == 0):
            self._hasCounterChild = False
        return True

    def showChildren(self) -> None:
        """
        Shows all children nodes of current node, with their ID. Function only shows children on Terminal.

        :return:
        """
        # Display all the children labels and IDs of current node.
        print("Current node:")
        print(self.label, "(with ID:", self.id, ")")
        print("")
        print("Children node(s):")
        for child in self.children:
            if child.leafNode:
                print(child.label, " (*)", "(with ID:", child.id, ")")
            else:
                print(child.label, "(with ID:", child.id, ")")
        if self.leafNode:
            # If no children, print the following.
            print("[No Children]")

    def changeRefinement(self) -> None:
        """
        Changes refinement of current node. No returns, since this operations cannot lead to an error.

        :return:
        """
        # Change refinement of current node.
        if self.refinement == 0:
            self.refinement = 1
        elif self.refinement == 1:
            self.refinement = 0

    def getNodeInformation(self) -> List[Union[Tuple[str, int, int, list, str, int], Tuple[str, int, int, list, None, int]]]:
        """
        Retrieve label, type, refinement, children, parent and ID in a list.

        :return: `list`: Contains label, type, refinement, children, parent and ID respectively.
        """
        # Retrieve label, type, refinement, parent and ID in a list.
        childLabels = []
        for i in range(len(self.children)):
            childLabels.append(self.children[i].label)
        if self.parent != None:
            return self.label, self.type, self.refinement, childLabels, self.parent.label, self.id
        else:
            return self.label, self.type, self.refinement, childLabels, None, self.id

    def changeType(self) -> bool:
        """
        Changes type of current node.

        :return: `bool`: `True` when successful, `False` otherwise.
        """
        # If current type is attack, change to defense. If current type is defense, change to attack.
        if self.type == 0:
            self.type = 1
        else:
            self.type = 0
        oneCounterChild = False # Parent is allowed to have at most one counter child.
        for child in self.children:
            # For all children, check if current child has counter. If not, set oneCounterChild to True. If yes, countermeasure prevention
            # prohibits from this node changing type.
            if self.type == 0:
                if child.type == 1 and not oneCounterChild:
                    oneCounterChild = True
                elif child.type == 1 and oneCounterChild:
                    if self.type == 0:
                        self.type = 1
                    else:
                        self.type = 0
                    return False
            else:
                if child.type == 0 and not oneCounterChild:
                    oneCounterChild = True
                elif child.type == 0 and oneCounterChild:
                    if self.type == 0:
                        self.type = 1
                    else:
                        self.type = 0
                    return False
        # If type is changed successfully, clear full countermeasure prevention and reset it all in the tree so that
        # this type change is registered in the total ADT countermeasure situation.
        self.root[1]._clearCountermeasurePrevention()
        self.root[1]._checkAllCountermeasurePrevention()
        return True

    def changeLabel(self, newLabel: str) -> bool:
        """
        Changes label of current node to provided label.

        :param newLabel: Contains the new label of the node.
        :return: `bool`: `True` when successful, `False` otherwise.
        """
        # Try changing the label. If something goes wrong, return False.
        try:
            self.label = newLabel
            return True
        except:
            return False

    def currentNodeToChildNode(self, idChild: int) -> Union['ADT', bool]:
        """
        Changes the current node that can be worked on to the child node with the given ID.

        :param idChild: `idChild`: ID of the child node.
        :return: `ADT`/`None`: `ADT` object of child node when child node with given ID exists, `None` otherwise.
        """
        # Check if given ID is indeed in the current node's children.
        try:
            for child in self.children:
                if int(idChild) == child.id:
                    # If yes, return this child.
                    return child
        except:
            # If not, return False.
            return False

    def currentNodeToParentNode(self) -> Union['ADT', bool]:
        """
        Changes the current node that can be worked on to the parent node.

        :return: `ADT`/`None`: `ADT` object of parent node when parent node is available, `None` if node has no parent (f.e. root node).
        """
        # If we are not in the root node, go to the parent node. Otherwise, return False.
        if (self.parent != 0):
            return self.parent
        else:
            return False


    def _checkCounterMeasure(self, child: 'ADT') -> bool:
        """
        Checks if the countermeasure prevention rules are kept.

        :param child: Child between which and the parent node countermeasure prevention must be checked.
        :return: `bool`: `True` if countermeasure prevention rules are kept, `False` otherwise.

        Note: *Helper function, users should not call this function themselves. Used in member functions: `addChild(...)` and `_clearCountermeasurePrevention(...)`.*
        """
        # If this node already has a counter, and new child is also a counter child, prohibit action and return False.
        if (self._hasCounterChild and self.type == 1 and child.type == 0) or (self._hasCounterChild and self.type == 0 and child.type == 1):
            return False
        # If this node does not have a counter child yet, set counter to True.
        if (self.type == 0 and child.type == 1) or (self.type == 1 and child.type == 0):
            self._hasCounterChild = True
        # Countermeasure prevention is checked! Return True.
        return True

    def _clearCountermeasurePrevention(self) -> None:
        """
        Clears Countermeasure Prevention. Resets `_hasCounterChild` variables of all nodes in ADT to `False`.
        :return:

        Note: *Helper function, users should not call this function themselves. Used in member function: `_checkAllCountermeasurePrevention(self)`.*
        """
        # For all children in Depth-First Search, recursively call this function. Set counter variables to False.
        for i in range(len(self.children)):
            if len(self.children) > 0:
                self.children[i]._clearCountermeasurePrevention()
                self._hasCounterChild = False

    def _checkAllCountermeasurePrevention(self) -> bool:
        """
        Checks Countermeasure Prevention after uploading a new ADT in XML format, removes it and redetermines it. Needs to be done manually to prevent Countermeasure Prevention glitches.

        :return: `bool`: `True` if successful, `False` otherwise.
        
        Note: *Helper function, users should not call this function themselves. Used in member function: `_checkAllCountermeasurePrevention(...)`.* 
        """
        # For all children in Depth-First Search, recursively call this function.
        for i in range(len(self.children)):
            if len(self.children) > 0:
                if not self.children[i]._checkAllCountermeasurePrevention():
                    return False
                if not self._checkCounterMeasure(self.children[i]):
                    # If Countermeasure Prevention prohibits action, return False.
                    return False
        # Countermeasure Prevention over whole tree is successful! Return True.
        return True

    # XML Operations - ADTXML.py
    def importXML(self, fileName: str = "", manual: bool = True) -> Tuple[Union['ADT', bool], int]:
        """
        Imports a new ADT with `self` as the root node `ADT` object in **XML class**. Filename can either be given as parameter or interactively via Terminal (to allow for dynamical file import) and should **not contain .xml**.

        :param fileName: Filename of the XML file, **excluding** .xml.
        :param manual: `True` if you want to give filename interactively, `False` if you want the process to go automatic, without human input needed.
        :return: `ADT`/`bool`, `int`: `ADT` object and `0` of root node of ADT if successful, `False` and `1` if file not found, `False` and `2` if imported ADT does not comply with countermeasure prevention.
        """
        XMLResult = self._XML._importXML(fileName, manual)
        if not XMLResult:
            return False, 1
        elif not XMLResult._checkAllCountermeasurePrevention():
            return False, 2
        return XMLResult, 0

    def exportXML(self, fileName: str = "", manual: bool = True) -> bool:
        """
        Exports the current ADT with `self` as the root node `ADT` object in **XML class**. User has to interactively give a filename for the exported file in the terminal.

        :param fileName: Filename of the XML file, **excluding** .xml.
        :param manual: `True` if you want to give filename interactively, `False` if you want the process to go automatic, without human input needed.
        :return: `bool`: `True` if exporting was successful, `False` otherwise.
        """
        if not self._XML._exportXML(fileName, manual):
            return False
        return True

    def _exportChild(self, root: minidom.Document, nodeChild: minidom.Element) -> bool:
        """
        Recursively adds children to the XML structure.

        :param root: Root node, `Document` object of the ADT that is exported.
        :param nodeChild: Children of this node, `ADT` objects, will be appended to this `Element` object.
        :return: `bool`: `True` when exporting children concluded successfully, `False` otherwise.

        Note: *Helper function, users should not call this function themselves. Used in member function: `exportXML(...)`.*
        """
        if self._XML._exportChild(root, nodeChild):
            return True
        return False

    def _determineLeafNodes(self) -> None:
        """
        Determines all the leaf nodes in the current ADT structure by setting the variable `self.leafNode` of all `ADT` objects of leaf nodes to `True`. No returns.

        :return:

        Note: *Helper function, users should not call this function themselves. Used in member function: `importXML(...)`.*
        """
        # Go through all nodes recursively.
        for i in range(len(self.children)):
            if len(self.children) > 0:
                self.children[i]._determineLeafNodes()
        if not self.children:
            # If this node has no children, then this is a leaf node.
            self.leafNode = True

    # ADT Satisfiability - ADTSatisfiability.py
    def determineSatisfiability(self) -> bool:
        """
        Determines satisfiability of the ADT, with the node it is run on as the root node.

        :return: `bool`: `True` if ADT is satisfiable, `False` if ADT is not satisfiable.
        """
        satResult = self._Satisfiability._determineSatisfiability()
        if satResult:
            self.root[1].count += 1
            return True
        else:
            self.root[1].count += 1
            return False

    # Generate ADTerm - GenerateADTerm.py
    def generateADTerm(self) -> str:
        """
        Generates ADTerm of the ADT, with the node it is run on as the root node.

        :return: `str`: ADTerm of the ADT in string format.
        """
        return self._GenADTerm._generateADTerm()

    # Compare ADTerms - CompareADTerms.py
    def compareADTerms(self, ADTerm1: str, ADTerm2: str) -> Union[Tuple[bool, int], Tuple[bool, None]]:
        """
        Determines whether two ADTerms are equivalent to each other by comparing the two. Includes check with equivalence threshold using the Levenshtein Distance.

        :param ADTerm1: ADTerm to compare with, generated by `generateADTerm(...)`.
        :param ADTerm2: ADTerm to compare with, generated by `generateADTerm(...)`.
        :return: `bool`, `int`/`None`: `True` and total Levenshtein Distance (`Int`) if equivalent, `False` and `None` otherwise.
        """
        resultComp = self._CompADTerms._compareADTerms(ADTerm1, ADTerm2)
        if not resultComp[0]:
            return False, None
        return True, resultComp[1]

    def changeLevenshteinErrorMargin(self, newValue: float) -> bool:
        """
        Changes Levenshtein Distance equivalence threshold value if new value is between 0.0 and 1.0.

        :param newValue: New value of Levenshtein Distance equivalence threshold.
        :return: `bool`: `True` if value changed successfully, `False` otherwise.
        """
        if not self._CompADTerms._changeLevenshteinErrorMargin(newValue):
            return False
        return True


    def generateStatistics(self, attackNodes: int = 0, defenseNodes: int = 0, leafNodes: int = 0, ORRefinements: int = 0, ANDRefinements: int = 0, counterMeasures: int = 0) -> Tuple[int, int, int, int, int, int]:
        """
        Generates statistics of ADT, including the amount of attack nodes, the amount of defense nodes, the amount of leaf nodes, the amount of OR refinements, the amount of AND refinements and the amount of countermeasures.

        :param attackNodes: Amount of attack nodes. Should not be used by user, unless a custom amount should be added. Is necessary for the recursion.
        :param defenseNodes: Amount of defense nodes. Should not be used by user, unless a custom amount should be added. Is necessary for the recursion.
        :param leafNodes: Amount of leaf nodes. Should not be used by user, unless a custom amount should be added. Is necessary for the recursion.
        :param ORRefinements: Amount of OR refinements. Should not be used by user, unless a custom amount should be added. Is necessary for the recursion.
        :param ANDRefinements: Amount of AND refinements. Should not be used by user, unless a custom amount should be added. Is necessary for the recursion.
        :param counterMeasures: Amount of countermeasures. Should not be used by user, unless a custom amount should be added. Is necessary for the recursion.
        :return: `list`: `int` values of the amount of attack nodes, defense nodes, leaf nodes, OR refinements, AND refinements and countermeasures respectively.
        """
        # All statistics are given as arguments of this function (default value is 0, function can be called without all those arguments).
        for i in range(len(self.children)):
            # For the children of the current node.
            if len(self.children) > 0:
                # If this node has children.
                # Go into this function recursively and let the return be all the statistics. We give the currently known statistics to the children (this way, in Depth First Search, f.e. the second child already has the statistics of the first child saved).
                attackNodes, defenseNodes, leafNodes, ORRefinements, ANDRefinements, counterMeasures = self.children[i].generateStatistics(attackNodes, defenseNodes, leafNodes, ORRefinements, ANDRefinements, counterMeasures)
                # From this point on, all statistics are collected Depth-First Search.

                # Determine type of the current node.
                if self.children[i].type == 0:
                    attackNodes += 1
                else:
                    defenseNodes += 1

                # Determine refinement of the current node (if this node is not a leafnode and has more than 1 child).
                if not self.children[i].leafNode and len(self.children[i].children) > 1:
                    if self.children[i].refinement == 0:
                        ORRefinements += 1
                    else:
                        ANDRefinements += 1

                # Determine leaf nodes.
                elif self.children[i].leafNode:
                    leafNodes += 1

                # Determine counter measure.
                if self.children[i].type != self.type:
                    counterMeasures += 1

        # For the root node.
        if self.root[0]:
            # Determine type of root node.
            if self.type == 0:
                attackNodes += 1
            else:
                defenseNodes += 1

            # Determine refinement of root node (if root node has more than 1 child).
            if self.refinement == 0 and len(self.children) > 1:
                ORRefinements += 1
            elif self.refinement == 1 and len(self.children) > 1:
                ANDRefinements += 1

            # Determine if root node has Counter Measure.
            for j in range(len(self.children)):
                if self.children[j].type != self.type:
                    counterMeasures += 1

        # Return current statistics. These statistics are given to (and updated by) the parent node.
        return attackNodes, defenseNodes, leafNodes, ORRefinements, ANDRefinements, counterMeasures

# Function initADT creates an ADT Root node and returns it to the user. This is not a member function of the ADT class.
def initADT(type: int, refinement: int, label: str) -> 'ADT':
    """
    Initializes an ADT with as root node a node with the given type, refinement and label by calling the constructor of
    class `ADT`. Returns the root node to the user, on which memberfunctions of the class `ADT` can be performed. Since
    this function is located in `ADT.py`, the location of this function must be given when calling it. Therefore, the
    prefix `ADT` when calling the function is required.
    :param type: Type of the root node (`0` is attack, `1` is defense).
    :param refinement: Refinement of the root node (`0` is disjunctive (OR), `1` is conjunctive (AND)).
    :param label: Label of the root node.
    :return: `ADT` object: Root node of the newly created ADT.
    """
    id = 0 # Initialize Root node with ID 0.
    parent = None # Root node does not have a parent.
    root = [True, None]
    rootNode = ADT(id, type, refinement, label, parent, root) # Create Root node.
    rootNode.root[1] = rootNode # Set newly created node as Root node.
    rootNode.usedIDs.append(id) # Append this ID to usedID list.
    rootNode.leafNode = True
    return rootNode




