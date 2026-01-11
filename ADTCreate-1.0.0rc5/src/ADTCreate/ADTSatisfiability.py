# Developed by Jayme Hebinck, for bachelor thesis Leiden University.
# This file is part of the ADT Create project, found on: https://github.com/MainJay/ADTPythonLibrary/

from . import ADT

class _Satisfiability:
    """
    Contains functions regarding Satisfiability operations on `ADT` objects.
    """
    def __init__(self, ADT: 'ADT.ADT') -> None:
        """
        Links `_Satisfiability` object to `ADT` object. `_Satisfiability` object is initialized when `ADT` object is initialized.
        :param ADT: Links `ADT` object to this `_Satisfiability` object, to make sure these two objects are linked correctly.
        """
        self.ADT = ADT

    def _determineSatisfiability(self) -> bool:
        """
        Determines satisfiability of the ADT, with the node is it run on as the root node.

        :return: `bool`: `True` if ADT is satisfiable, `False` if ADT is not satisfiable.

        Note: *This function is called from the `ADT` class using `determineSatisfiability(...)`, and should therefore not be called from the `_Satisfiability` class.*
        """
        if not self.ADT.leafNode and len(self.ADT.children) > 1:
            # For all the nodes that are not leaf nodes and have more than one child.
            if self.ADT.refinement == 0:
                # For every child that has OR refinement.
                for i in range(len(self.ADT.children)):
                    # Either of these options is enough to prove there is at least one satisfiable path.
                    # Check if any direct counter child node for this node.
                    if self.ADT.type == self.ADT.children[i].type:
                        if self.ADT.children[i].determineSatisfiability():
                            for j in range(len(self.ADT.children)):
                                # print(self.ADT.children[j].label)
                                # If a direct counter child node is found, return False
                                if self.ADT.children[j].type != self.ADT.type:
                                    # print(self.ADT.children[j].label) # THIS PART IS GOOD
                                    if self.ADT.children[j].determineSatisfiability():
                                        return False
                            # If the type of the parent and child are the same, and the child is satisfiable, and there is
                            # no direct counter child node, then the parent is also satisfiable.
                            return True
                    elif self.ADT.type != self.ADT.children[i].type:
                        if not self.ADT.children[i].determineSatisfiability():
                            # If the child is a counter node, and it is not satisfiable, then the parent node is
                            # satisfiable.
                            return True
                return False
            else:
                # For every child that has AND refinement.
                for i in range(len(self.ADT.children)):
                    # Either of these options is enough to prove this node is not satisfiable.
                    # Skip the other children, not necessary to check those.
                    if not self.ADT.children[i].determineSatisfiability():
                        # If the child node is not satisfiable, then the parent is not satisfiable.
                        return False
                    elif self.ADT.type != self.ADT.children[i].type:
                        # If the child node is satisfiable, but the child is a counter node, then the parent is not
                        # satisfiable.
                        return False
                return True
        elif not self.ADT.leafNode and len(self.ADT.children) == 1:
            # For all the nodes that are not leaf nodes and have one child.
            if self.ADT.type == self.ADT.children[0].type:
                # If the parent and child nodes are of same type, and child is satisfiable, then so is the parent.
                if self.ADT.children[0].determineSatisfiability():
                    return True
            else:
                # If the parent and child nodes are not of the same type, and the child is not satisfiable, then the
                # parent is.
                # print(self.ADT.label, self.ADT.children[0].label)
                if not self.ADT.children[0].determineSatisfiability():
                    return True
            return False
        else:
            # We assume all basic actions (leaf nodes) are satisfiable.
            return True