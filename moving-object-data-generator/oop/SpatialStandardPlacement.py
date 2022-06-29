import numpy as np

from oop.StandardInitiation import StandardInitiation


class SpatialStandardPlacement:
    def __init__(self):
        self.features_ids = np.array([], dtype=np.int32)
        self.features_instances_ids = np.array([], dtype=np.int32)
        self.x = np.array([], dtype=np.float64)
        self.y = np.array([], dtype=np.float64)

    def place(self, si: StandardInitiation):

        # delete previous placement of the objects
        self.features_ids = np.array([], dtype=np.int32)
        self.features_instances_ids = np.array([], dtype=np.int32)
        self.x = np.array([], dtype=np.float64)
        self.y = np.array([], dtype=np.float64)

        # take parameters which were used in standard initiation
        sp = si.standard_parameters

        collocation_start_feature_id = 0
        collocation_features_instances_counters = np.zeros(shape=si.collocation_features_sum, dtype=np.int32)

        # generate data of every co-location in given time frame
        for i_colloc in range(sp.n_colloc * sp.m_overlap):

            # get the features ids of current co-location
            collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + si.collocation_lengths[i_colloc])
            collocation_features[-1] += i_colloc % sp.m_overlap
            print("collocation_features=%s" % str(collocation_features))

            # calculate total number of all co-location feature instances
            collocation_features_instances_sum = si.collocations_instances_counts[i_colloc] * si.collocation_lengths[i_colloc]

            # generate vector of x coordinates of all the consecutive instances
            collocation_features_instances_x = np.random.randint(low=si.area_in_cell_dim, size=(si.collocations_instances_counts[i_colloc] - 1) // sp.m_clumpy + 1)
            collocation_features_instances_x *= sp.cell_size
            collocation_features_instances_x = collocation_features_instances_x.astype(dtype=np.float64)
            collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=sp.m_clumpy)[:si.collocations_instances_counts[i_colloc]]
            collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=si.collocation_lengths[i_colloc])
            collocation_features_instances_x += np.random.uniform(high=sp.cell_size, size=collocation_features_instances_sum)

            # generate vector of y coordinates of all the consecutive instances
            collocation_features_instances_y = np.random.randint(low=si.area_in_cell_dim, size=(si.collocations_instances_counts[i_colloc] - 1) // sp.m_clumpy + 1)
            collocation_features_instances_y *= sp.cell_size
            collocation_features_instances_y = collocation_features_instances_y.astype(dtype=np.float64)
            collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=sp.m_clumpy)[:si.collocations_instances_counts[i_colloc]]
            collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=si.collocation_lengths[i_colloc])
            collocation_features_instances_y += np.random.uniform(high=sp.cell_size, size=collocation_features_instances_sum)

            # generate vector of features ids of all the consecutive instances
            collocation_features_ids = np.tile(A=collocation_features, reps=si.collocations_instances_counts[i_colloc])

            # generate vector of features instances ids of all the consecutive instances
            collocation_features_instances_ids = np.arange(
                start=collocation_features_instances_counters[collocation_start_feature_id],
                stop=collocation_features_instances_counters[collocation_start_feature_id] + si.collocations_instances_counts[i_colloc]
            )
            collocation_features_instances_ids = np.tile(A=collocation_features_instances_ids, reps=(si.collocation_lengths[i_colloc] - 1, 1))
            collocation_features_instances_ids = np.concatenate((
                collocation_features_instances_ids,
                np.arange(si.collocations_instances_counts[i_colloc]).reshape((1, si.collocations_instances_counts[i_colloc]))
            ))
            collocation_features_instances_ids = collocation_features_instances_ids.T.flatten()

            # remember data of current co-location features
            self.features_ids = np.concatenate((self.features_ids, collocation_features_ids))
            self.features_instances_ids = np.concatenate((self.features_instances_ids, collocation_features_instances_ids))
            self.x = np.concatenate((self.x, collocation_features_instances_x))
            self.y = np.concatenate((self.y, collocation_features_instances_y))

            # increase counts of created instances of the co-location features which occurred in current co-location
            collocation_features_instances_counters[collocation_features] += si.collocations_instances_counts[i_colloc]

            # change starting feature of next co-location according to the m_overlap parameter value
            if (i_colloc + 1) % sp.m_overlap == 0:
                collocation_start_feature_id += si.collocation_lengths[i_colloc] + sp.m_overlap - 1

        # generate data of every co-location noise feature in given time frame
        # generate vectors of x and y coordinates of all the consecutive instances of co-location noise features
        collocation_noise_features_instances_x = np.random.uniform(high=sp.area, size=si.collocation_noise_features_instances_sum)
        collocation_noise_features_instances_y = np.random.uniform(high=sp.area, size=si.collocation_noise_features_instances_sum)

        # remember data of co-location noise features
        self.features_ids = np.concatenate((self.features_ids, si.collocation_noise_features_ids))
        self.features_instances_ids = np.concatenate((self.features_instances_ids, si.collocation_noise_features_instances_ids))
        self.x = np.concatenate((self.x, collocation_noise_features_instances_x))
        self.y = np.concatenate((self.y, collocation_noise_features_instances_y))

        # generate additional noise features if they are requested in given time frame
        if sp.ndf > 0:
            # generate vectors of x and y coordinates of all the consecutive instances of additional noise features
            additional_noise_features_instances_x = np.random.uniform(high=sp.area, size=sp.ndfn)
            additional_noise_features_instances_y = np.random.uniform(high=sp.area, size=sp.ndfn)

            # remember data of additional noise features
            self.features_ids = np.concatenate((self.features_ids, si.additional_noise_features_ids))
            self.features_instances_ids = np.concatenate((self.features_instances_ids, si.additional_noise_features_instances_ids))
            self.x = np.concatenate((self.x, additional_noise_features_instances_x))
            self.y = np.concatenate((self.y, additional_noise_features_instances_y))
