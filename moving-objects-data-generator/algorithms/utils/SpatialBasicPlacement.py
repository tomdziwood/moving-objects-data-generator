import numpy as np

from algorithms.initiation.BasicInitiation import BasicInitiation


class SpatialBasicPlacement:
    def __init__(self):
        self.features_instances_coor: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)

    def place(self, bi: BasicInitiation):

        # take parameters which were used in basic initiation
        bp = bi.basic_parameters

        # vectorized method of generating all features instances coordinates - with the awareness of the m_clumpy parameter
        collocations_clumpy_instances_coor = np.random.randint(low=bi.area_in_cell_dim, size=(bi.collocations_clumpy_instances_global_sum, 2))
        collocations_clumpy_instances_coor *= bp.cell_size
        collocations_clumpy_instances_coor = collocations_clumpy_instances_coor.astype(dtype=np.float64)
        self.features_instances_coor = collocations_clumpy_instances_coor[bi.collocations_clumpy_instances_global_ids]
        self.features_instances_coor += np.random.uniform(high=bp.cell_size, size=self.features_instances_coor.shape)
