import numpy as np

from algorithms.initiation.BasicInitiation import BasicInitiation


class SpatialStandardPlacement:
    def __init__(self,
                 bi: BasicInitiation = BasicInitiation(),
                 collocations_instances_number_spatial_prevalence_threshold: np.ndarray = np.array([], dtype=np.int32)):

        self.bi: BasicInitiation = bi
        self.collocations_instances_number_spatial_prevalence_threshold: np.ndarray = collocations_instances_number_spatial_prevalence_threshold
        self.collocations_instances_spatial_prevalent_flags: np.ndarray = np.array([], dtype=bool)
        self.features_instances_coor: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)

    def place(self, collocations_spatial_prevalence_flags):

        # take parameters which were used in basic initiation
        bp = self.bi.basic_parameters

        # determine the number of the co-locations instances which actually creates co-location in the current time frame
        collocations_spatial_prevalent_instances_number = np.zeros(shape=self.bi.collocations_sum, dtype=np.int32)
        collocations_spatial_prevalent_instances_number[np.logical_not(collocations_spatial_prevalence_flags)] = np.random.randint(
            low=0,
            high=self.collocations_instances_number_spatial_prevalence_threshold[np.logical_not(collocations_spatial_prevalence_flags)]
        )
        collocations_spatial_prevalent_instances_number[collocations_spatial_prevalence_flags] = np.random.randint(
            low=self.collocations_instances_number_spatial_prevalence_threshold[collocations_spatial_prevalence_flags],
            high=self.bi.collocation_instances_counts[collocations_spatial_prevalence_flags] + 1
        )

        # ---begin--- create boolean vector which tells if the given co-locations instance occurs in the current time frame
        collocation_instances_counts_cumsum = np.cumsum(self.bi.collocation_instances_counts)

        shuffled_values = np.repeat(
            a=self.bi.collocation_instances_counts - collocation_instances_counts_cumsum,
            repeats=self.bi.collocation_instances_counts
        ) + np.arange(1, self.bi.collocations_instances_sum + 1)

        ind_begin = np.concatenate(([0], collocation_instances_counts_cumsum[: -1]))

        [np.random.shuffle(shuffled_values[ind_begin[i]: collocation_instances_counts_cumsum[i]]) for i in range(self.bi.collocations_sum)]

        self.collocations_instances_spatial_prevalent_flags = shuffled_values <= np.repeat(a=collocations_spatial_prevalent_instances_number, repeats=self.bi.collocation_instances_counts)
        # ----end---- create boolean vector which tells if the given co-locations instance occurs in the current time frame

        # expand co-locations' instances' flags into features' instances' flags
        features_instances_spatial_prevalent_flags = np.repeat(
            a=self.collocations_instances_spatial_prevalent_flags,
            repeats=self.bi.collocations_instances_global_ids_repeats[:self.bi.collocations_instances_sum]
        )

        # initialize features' instances' coordinates as if there were no co-locations' instances occurrences at all
        self.features_instances_coor = np.random.uniform(high=self.bi.area_in_cell_dim * bp.cell_size, size=(self.bi.features_instances_sum, 2))

        # initialize features' instances' coordinates as if there occurred every co-locations' instance - with the awareness of the m_clumpy parameter
        collocations_clumpy_instances_coor_all_collocations_instances_occured = np.random.randint(low=self.bi.area_in_cell_dim, size=(self.bi.collocations_clumpy_instances_global_sum, 2))
        collocations_clumpy_instances_coor_all_collocations_instances_occured *= bp.cell_size
        collocations_clumpy_instances_coor_all_collocations_instances_occured = collocations_clumpy_instances_coor_all_collocations_instances_occured.astype(dtype=np.float64)
        features_instances_coor_all_collocations_instances_occured = collocations_clumpy_instances_coor_all_collocations_instances_occured[self.bi.collocations_clumpy_instances_global_ids]
        features_instances_coor_all_collocations_instances_occured += np.random.uniform(high=bp.cell_size, size=features_instances_coor_all_collocations_instances_occured.shape)

        # mix features' instances' coordinates according to the 'features_instances_spatial_prevalent_flags'
        self.features_instances_coor[:self.bi.collocation_features_instances_sum][features_instances_spatial_prevalent_flags] = \
            features_instances_coor_all_collocations_instances_occured[:self.bi.collocation_features_instances_sum][features_instances_spatial_prevalent_flags]
