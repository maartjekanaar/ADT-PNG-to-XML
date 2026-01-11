# Developed by Jayme Hebinck, for bachelor thesis Leiden University.
# This file is part of the ADT Create project, found on: https://github.com/MainJay/ADTPythonLibrary/.

class MSList(list):
    """
    Implementation of multisets, based on `list`. `list` is edited so that the order of elements do not matter when two objects of this class are compared.
    """
    # This class inherits the built-in list class of Python. It then changes the __eq__ function, so that comparison is
    # not based on the order of elements in the list, but rather on if this item even is in this MSList.
    # A list like this, where order does not matter, is called a Multi-Set. Therefore, this class is called the
    # MSList.
    def __eq__(self, other: 'MSList') -> bool:
        """
        This equal function has been modified slightly, so that order of elements does not matter when two objects of `MSList` are compared.

        :param other: Other multiset that this multiset is compared to.
        :return: `bool`: `True` when multisets are equal, `False` otherwise.
        """
        for item in self:
            if item not in other:
                return False
        return True



