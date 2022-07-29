import numpy as np

from oop.BasicInitiation import BasicInitiation


class SpatialBasicPlacement:
    def __init__(self):
        self.x = np.array([], dtype=np.float64)
        self.y = np.array([], dtype=np.float64)

    def place(self, bi: BasicInitiation):

        # delete previous placement of the objects
        self.x = np.array([], dtype=np.float64)
        self.y = np.array([], dtype=np.float64)

        # take parameters which were used in basic initiation
        bp = bi.basic_parameters

        # generate data of every co-location in given time frame
        for i_colloc in range(bp.n_colloc * bp.m_overlap):

            # calculate total number of all co-location feature instances
            collocation_features_instances_sum = bi.collocation_instances_counts[i_colloc] * bi.collocation_lengths[i_colloc]

            # generate vector of x coordinates of all the consecutive instances
            collocation_features_instances_x = np.random.randint(low=bi.area_in_cell_dim, size=(bi.collocation_instances_counts[i_colloc] - 1) // bp.m_clumpy + 1)
            collocation_features_instances_x *= bp.cell_size
            collocation_features_instances_x = collocation_features_instances_x.astype(dtype=np.float64)
            collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=bp.m_clumpy)[:bi.collocation_instances_counts[i_colloc]]
            collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=bi.collocation_lengths[i_colloc])
            collocation_features_instances_x += np.random.uniform(high=bp.cell_size, size=collocation_features_instances_sum)

            # generate vector of y coordinates of all the consecutive instances
            collocation_features_instances_y = np.random.randint(low=bi.area_in_cell_dim, size=(bi.collocation_instances_counts[i_colloc] - 1) // bp.m_clumpy + 1)
            collocation_features_instances_y *= bp.cell_size
            collocation_features_instances_y = collocation_features_instances_y.astype(dtype=np.float64)
            collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=bp.m_clumpy)[:bi.collocation_instances_counts[i_colloc]]
            collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=bi.collocation_lengths[i_colloc])
            collocation_features_instances_y += np.random.uniform(high=bp.cell_size, size=collocation_features_instances_sum)

            # remember data of current co-location features
            self.x = np.concatenate((self.x, collocation_features_instances_x))
            self.y = np.concatenate((self.y, collocation_features_instances_y))

        # generate data of every co-location noise feature in given time frame
        # generate vectors of x and y coordinates of all the consecutive instances of co-location noise features
        collocation_noise_features_instances_x = np.random.uniform(high=bp.area, size=bi.collocation_noise_features_instances_sum)
        collocation_noise_features_instances_y = np.random.uniform(high=bp.area, size=bi.collocation_noise_features_instances_sum)

        # remember data of co-location noise features
        self.x = np.concatenate((self.x, collocation_noise_features_instances_x))
        self.y = np.concatenate((self.y, collocation_noise_features_instances_y))

        # generate additional noise features if they are requested in given time frame
        if bp.ndf > 0:
            # generate vectors of x and y coordinates of all the consecutive instances of additional noise features
            additional_noise_features_instances_x = np.random.uniform(high=bp.area, size=bp.ndfn)
            additional_noise_features_instances_y = np.random.uniform(high=bp.area, size=bp.ndfn)

            # remember data of additional noise features
            self.x = np.concatenate((self.x, additional_noise_features_instances_x))
            self.y = np.concatenate((self.y, additional_noise_features_instances_y))
