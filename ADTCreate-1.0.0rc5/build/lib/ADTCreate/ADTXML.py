# Developed by Jayme Hebinck, for bachelor thesis Leiden University.
# This file is part of the ADT Create project, found on: https://github.com/MainJay/ADTPythonLibrary/.

from xml.dom import minidom
from . import ADT

from typing import List, Tuple, Union, Optional

class _XML:
    """
    Contains functions regarding XML operations on `ADT` objects.
    """
    def __init__(self, ADT: 'ADT.ADT') -> None:
        """
        Links `_XML` object to `ADT` object. `_XML` object is initialized when `ADT` object is initialized.
        :param ADT: Links `ADT` object to this `_XML` object, to make sure these two objects are linked correctly.
        """
        self.ADT = ADT

    def _exportChild(self, root: minidom.Document, nodeChild: minidom.Element) -> bool:
        """
        Recursively adds children to the XML structure.

        :param root: Root node, `Document` object of the ADT that is exported.
        :param nodeChild: Children of this node, `ADT` objects, will be appended to this `Element` object.
        :return: `bool`: `True` when exporting children concluded successfully, `False` otherwise.

        Note: *Helper function, users should not call this function themselves. Used in member function: `exportXML(...)`.*

        Note: *This function is called from the `ADT` class using `exportXML(...)`, and should therefore not be called from the `_XML` class.*
        """
        # For all children recursively, add the nodes to the .xml.
        for i in range(len(self.ADT.children)):
            nodeChild2 = root.createElement('node')
            # Add refinement and type.
            if (self.ADT.children[i].refinement == 0):
                nodeChild2.setAttribute('refinement', 'disjunctive')
            else:
                nodeChild2.setAttribute('refinement', 'conjunctive')
            if (self.ADT.type != self.ADT.children[i].type):
                nodeChild2.setAttribute('switchRole', 'yes')
            labelChild2 = root.createElement('label')
            labelChildText2 = root.createTextNode(self.ADT.children[i].label)
            # Add the new child to the existing .xml.
            nodeChild.appendChild(nodeChild2)
            nodeChild2.appendChild(labelChild2)
            labelChild2.appendChild(labelChildText2)
            self.ADT.children[i]._exportChild(root, nodeChild2) # Recursively call with next node.
        return True

    def _exportXML(self, filename: str = "", manual: bool = True) -> bool:
        """
        Exports the current ADT with `self` as the root node `ADT` object in **XML class**. User has to interactively give a filename for the exported file in the terminal.

        :param filename: Filename of the XML file, **excluding** .xml.
        :param manual: `True` if you want to give filename interactively, `False` if you want the process to go automatic, without human input needed.
        :return: `bool`: `True` if exporting was successful, `False` otherwise.

        Note: *This function is called from the `ADT` class using `exportXML(...)`, and should therefore not be called from the `_XML` class.*
        """
        # Initialize needed variables.
        root = minidom.Document()
        xml = root.createElement('adtree')
        root.appendChild(xml)
        nodeChild = root.createElement('node')
        # For the root node, set refinement and type for in the .xml.
        if (self.ADT.refinement == 0):
            nodeChild.setAttribute('refinement', 'disjunctive')
        else:
            nodeChild.setAttribute('refinement', 'conjunctive')
        if (self.ADT.type == 1):
            nodeChild.setAttribute('switchRole', 'yes')
        labelChild = root.createElement('label')
        labelChildText = root.createTextNode(self.ADT.label)
        # Append this to the .xml.
        xml.appendChild(nodeChild)
        nodeChild.appendChild(labelChild)
        labelChild.appendChild(labelChildText)
        # Now with the known .xml structure and the root already in the .xml, add the rest of the nodes to the .xml.
        self.ADT._exportChild(root, nodeChild)

        # Set .xml settings and ask user for output name.
        xmlContent = root.toprettyxml(indent="  ")
        if manual:
            filename = input("File output name (do not add an extension): ")
        filename = filename + ".xml"

        # Write file with given name to the file directory.
        with open(filename, "w") as f:
            f.write(xmlContent)

        return True

    def _importXML(self, fileName: str = "", manual: bool = True) -> Union['ADT.ADT', bool]:
        """
        Imports a new ADT with `self` as the root node `ADT` object. Filename can either be given as parameter or interactively via Terminal (to allow for dynamical file import) and should **not contain .xml**.

        :param fileName: Filename of the XML file, **excluding** .xml.
        :param manual: `True` if you want to give filename interactively, `False` if you want the process to go automatic, without human input needed.
        :return: `ADT`/`bool`: `ADT` object of root node of ADT if successful, `False` otherwise.

        Note: *This function is called from the `ADT` class using `importXML(...)`, and should therefore not be called from the `_XML` class.*
        """
        if manual:
            fileName = input("File input name (do not add an extension): ")
        try:
            pastusedIDs = self.ADT.usedIDs
            self.ADT.usedIDs.clear() # Resets used IDs if ADTs have been uploaded prior to this ADT.
            root = minidom.parse(fileName + ".xml")  # Import the .xml file
            adtree = root.firstChild  # Gain the first tag (adtree)
            children = adtree.getElementsByTagName("node")  # Retrieve all nodes in the .xml
            labels = adtree.getElementsByTagName("label")  # Retrieve all labels in the .xml
            depths = {}  # Initialize depths dictionary
            treeNodeStructure = {}  # Initialize treeNodeStructure dictionary
            treeLabelStructure = {}  # Initialize treeLabelStructure dictionary
            # Initialize needed variables for looping and identifying nodes.
            if not self.ADT.usedIDs:
                nodeID = 0
            else:
                nodeID = self.ADT.usedIDs[-1] + 1
            treeLoop = 0
            labelLoop = nodeID

            # Retrieve labels
            for label in labels:
                # Save labels to treeLabelStructure dictionary.
                treeLabelStructure[labelLoop] = label.firstChild.nodeValue
                labelLoop = labelLoop + 1

            # Retrieve depth
            depthLoopInit = 0
            deepest = 0
            for child in children:
                # Determine the depth of the nodes and save it in the depths dictionary.
                depth = 0
                while child.parentNode:
                    depth = depth + 1
                    child = child.parentNode
                    if depth > deepest:
                        deepest = depth
                depths[depthLoopInit] = depth
                depthLoopInit = depthLoopInit + 1

            depthParent = [0] * (deepest + 1)  # Initialize list of most recent parents of nodes on all depth levels in .xml

            depthLoop = 0  # Initialize new loop variable

            for child in children:
                # For all the nodes in this .xml
                # Retrieve refinement
                refNodeStr = child.getAttribute("refinement")
                if (refNodeStr == "disjunctive"):
                    # If refinement is set to 'disjunctive', this node has the OR refinement, which is type 0.
                    refNode = 0
                elif (refNodeStr == "conjunctive"):
                    # If refinement is set to 'conjunctive', this node has the AND refinement, which is type 1.
                    refNode = 1

                # Determine parent
                parentNode = 0  # The parent node, used in creating the ADT resulting from the import.
                if depthLoop != 0:
                    # As long as this is not the root.
                    if depths[depthLoop - 1] < depths[depthLoop]:
                        # If depth of previous node is lower than the current node, it is sure that the previous node
                        # is the parent node.
                        depthParent[depths[depthLoop]] = treeNodeStructure[depthLoop - 1]
                        parentNode = treeNodeStructure[depthLoop - 1]
                    elif depths[depthLoop - 1] == depths[depthLoop]:
                        # If depth of previous node is equal to the current node, it is sure that the previous node
                        # is the parent node, but the most recent parent's node depth of this node.
                        parentNode = treeNodeStructure[depthLoop - 1].parent
                    elif depths[depthLoop - 1] > depths[depthLoop]:
                        # If depth of previous node is higher than current node, we go back to the most recent node
                        # on this depth. The parent of this node is guaranteed also the parent of this node.
                        parentNode = depthParent[depths[depthLoop]]

                # Retrieve type
                typeNodeStr = child.getAttribute("switchRole")
                # If root, type is 0 (attack node)
                if parentNode is None or parentNode == 0:
                    typeNode = 0
                # If switchRole is present in the .xml, this node has the opposite type of the parent node.
                else:
                    if typeNodeStr == "yes":
                        typeNode = 1 - parentNode.type
                    else:
                        typeNode = parentNode.type

                # Using all gathered information, create ADT of this node.
                ADTInstance = _createADTInstance(nodeID, typeNode, refNode, treeLabelStructure[nodeID], parentNode,
                                       [False, None])
                ADTInstance.usedIDs.append(nodeID) # Fill usedIDs list to keep track of all used IDs.
                treeNodeStructure[treeLoop] = ADTInstance

                # Increment all needed variables for looping and identification.
                nodeID = nodeID + 1
                treeLoop = treeLoop + 1
                depthLoop = depthLoop + 1

            for node in range(treeLoop):
                # Create the list of children of every node in the ADTs.
                if (treeNodeStructure[node].parent != 0):
                    treeNodeStructure[node].parent.children.append(treeNodeStructure[node])
                    treeNodeStructure[node].root = (False, treeNodeStructure[0])
                else:
                    treeNodeStructure[node].parent = None # None to describe a node does not have a parent, instead of 0.
                    treeNodeStructure[node].root = [True, treeNodeStructure[0]]
            treeNodeStructure[0]._determineLeafNodes()
            return treeNodeStructure[
                0]  # Return the root of the tree as ADT object.
        except:
            self.ADT.usedIDs = pastusedIDs # If import process fails, set usedIDs back to usedIDs from previous ADT.
            return False # If file not found, then return False.


def _createADTInstance(ID: int, type: int, refinement: int, label: str, parent: Optional['ADT.ADT'], root: List[Tuple[bool, Optional['ADT.ADT']]]) -> 'ADT.ADT':
    """
    Creates an object of class `ADT`, with information based on XML import. Needed in XML importation process.

    :param ID: ID number of new `ADT` object.
    :param type: Type of new `ADT` object.
    :param refinement: Refinement of new `ADT` object.
    :param label: Label of new `ADT` object.
    :param parent: Parent node of new `ADT` object.
    :param root: Root node of ADT to which new `ADT` object is added.
    :return: `ADT` object: returned after constructor is called to create a new node.

    Note: *Helper function, users should not call this function themselves. Used in member function: `_importXML(...)`*
    """
    return ADT.ADT(ID, type, refinement, label, parent, root)
