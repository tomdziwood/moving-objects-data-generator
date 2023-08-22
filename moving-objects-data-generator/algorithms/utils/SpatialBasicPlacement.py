import numpy as np

from algorithms.initiation.BasicInitiation import BasicInitiation


class SpatialBasicPlacement:
    """
    The class of a spatial placement. Object of this class performs placement of all features instances and holds all data of this placement. The placement is performed
    according to the strategy of the `SpatioTemporalBasicGenerator` generator.

    Attributes
    ----------
    features_instances_coor : np.ndarray
        The array's size is equal to the number of features instances. The i-th value represents the coordinates of the location of the i-th feature instance.
        The locations are generated with the ``place`` method call.
    """

    def __init__(self):
        """
        Construct empty object of the `SpatialBasicPlacement` class.
        """

        self.features_instances_coor: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)

    def place(self, bi: BasicInitiation):
        """
        Perform placement of all features instances initiated by the object of `BasicInitiation` class. Every object has assigned a single location
        in two-dimensional area of the spatial framework.

        Parameters
        ----------
        bi : BasicInitiation
            The object of a `BasicInitiation` initiation. This object contains data of all objects for which the placement is performed.
        """

        # take parameters which were used in basic initiation
        bp = bi.basic_parameters

        # vectorized method of generating all features instances coordinates - with the awareness of the m_clumpy parameter
        collocations_clumpy_instances_coor = np.random.randint(low=bi.area_in_cell_dim, size=(bi.collocations_clumpy_instances_global_sum, 2))
        collocations_clumpy_instances_coor *= bp.cell_size
        collocations_clumpy_instances_coor = collocations_clumpy_instances_coor.astype(dtype=np.float64)
        self.features_instances_coor = collocations_clumpy_instances_coor[bi.collocations_clumpy_instances_global_ids]
        self.features_instances_coor += np.random.uniform(high=bp.cell_size, size=self.features_instances_coor.shape)
