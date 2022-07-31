import numpy as np

from oop.BasicParameters import BasicParameters


class BasicInitiation:
    """
    The class of a `SpatioTemporalBasicGenerator` initiation. Object of this class stores all initial data, which is required to generate spatio-temporal data
    in each time frame.

    Attributes
    ----------
    basic_parameters : BasicParameters
        The object of class `BasicParameters`, which holds all the required parameters of the `SpatioTemporalBasicGenerator` generator.

    collocations_instances_global_ids : np.ndarray
        The array's size is equal to the number of features' instances. The i-th value represents the global id of the co-location instance,
        to which the i-th feature instance belongs.

    collocations_instances_global_sum : int
        The number of all specified collocation global instances.

    collocations_instances_global_ids_repeats : np.ndarray
        The array's size is equal to the number of co-locations instances. The i-th value represents the number of features' instances,
        which belong to the i-th co-location instance.
    """

    def __init__(self):
        """
        Construct empty object of the `BasicInitiation` class.
        """

        self.basic_parameters: BasicParameters = BasicParameters()
        self.base_collocation_lengths: np.ndarray = np.array([], dtype=np.int32)
        self.collocation_lengths: np.ndarray = np.array([], dtype=np.int32)
        self.collocations_sum: int = 0
        self.collocation_instances_counts: np.ndarray = np.array([], dtype=np.int32)
        self.collocation_features_sum: int = 0
        self.collocation_features_instances_counts: np.ndarray = np.array([], dtype=np.int32)
        self.collocation_features_instances_sum: int = 0
        self.collocation_features_ids: np.ndarray = np.array([], dtype=np.int32)
        self.collocation_features_instances_ids: np.ndarray = np.array([], dtype=np.int32)
        self.area_in_cell_dim: int = 0
        self.collocation_noise_features_sum: int = 0
        self.collocation_noise_features: np.ndarray = np.array([], dtype=np.int32)
        self.collocation_noise_features_instances_sum: int = 0
        self.collocation_noise_features_instances_counts: np.ndarray = np.array([], dtype=np.int32)
        self.collocation_noise_features_ids: np.ndarray = np.array([], dtype=np.int32)
        self.collocation_noise_features_instances_ids: np.ndarray = np.array([], dtype=np.int32)
        self.additional_noise_features: np.ndarray = np.array([], dtype=np.int32)
        self.additional_noise_features_instances_counts: np.ndarray = np.array([], dtype=np.int32)
        self.additional_noise_features_ids: np.ndarray = np.array([], dtype=np.int32)
        self.additional_noise_features_instances_ids: np.ndarray = np.array([], dtype=np.int32)
        self.features_sum: int = 0
        self.features_ids: np.ndarray = np.array([], dtype=np.int32)
        self.features_instances_ids: np.ndarray = np.array([], dtype=np.int32)
        self.features_instances_sum: int = 0
        self.collocations_instances_global_ids: np.ndarray = np.array([], dtype=np.int32)
        self.collocations_instances_global_sum: int = 0
        self.collocations_instances_global_ids_repeats: np.ndarray = np.array([], dtype=np.int32)
        self.collocation_clumpy_instances_counts: np.ndarray = np.array([], dtype=np.int32)
        self.collocations_clumpy_instances_global_ids: np.ndarray = np.array([], dtype=np.int32)
        self.collocations_clumpy_instances_global_sum: int = 0
        self.collocations_clumpy_instances_global_ids_repeats: np.ndarray = np.array([], dtype=np.int32)

    def initiate(self, bp: BasicParameters = BasicParameters()):
        """
        Initiate required data to generate spatio-temporal data in each time frame.

        Parameters
        ----------
        bp: BasicParameters
            The object of class `BasicParameters`, which holds all the required parameters of the `SpatioTemporalBasicGenerator` generator.
            Its attributes will be used to initialize required data.
        """

        self.basic_parameters = bp

        # set random seed value
        if bp.random_seed is not None:
            np.random.seed(bp.random_seed)

        # determine length to each of the n_colloc basic co-locations with poisson distribution (lam=lambda_1)
        self.base_collocation_lengths = np.random.poisson(lam=bp.lambda_1, size=bp.n_colloc)
        self.base_collocation_lengths[self.base_collocation_lengths < 2] = 2
        print("base_collocation_lengths=%s" % str(self.base_collocation_lengths))

        # set final length to each of the co-locations according to the m_overlap parameter value
        if bp.m_overlap > 1:
            self.collocation_lengths = np.repeat(a=self.base_collocation_lengths + 1, repeats=bp.m_overlap)
        else:
            self.collocation_lengths = self.base_collocation_lengths
        print("collocation_lengths=%s" % str(self.collocation_lengths))

        # the total number of co-locations patterns
        self.collocations_sum = self.collocation_lengths.size
        print("collocations_sum=%s" % str(self.collocations_sum))

        # determine number of instances to each of the co-locations with poisson distribution (lam=lambda_2)
        self.collocation_instances_counts = np.random.poisson(lam=bp.lambda_2, size=self.collocations_sum)
        self.collocation_instances_counts[self.collocation_instances_counts == 0] = 1
        print("collocation_instances_counts=%s" % str(self.collocation_instances_counts))

        # determine the total number of features, which take part in co-locations
        self.collocation_features_sum = np.sum(self.base_collocation_lengths)
        if bp.m_overlap > 1:
            self.collocation_features_sum += bp.n_colloc * bp.m_overlap
        print("collocation_features_sum=%d" % self.collocation_features_sum)

        # prepare count of all instances of every i'th co-location feature
        self.collocation_features_instances_counts = np.zeros(shape=self.collocation_features_sum, dtype=np.int32)

        # prepare arrays of co-location features ids and instances ids
        self.collocation_features_ids = np.array([], dtype=np.int32)
        self.collocation_features_instances_ids = np.array([], dtype=np.int32)

        # gather data for each co-location
        collocation_start_feature_id = 0
        for i_colloc in range(self.collocations_sum):

            # get the features ids of current co-location
            collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + self.collocation_lengths[i_colloc])
            collocation_features[-1] += i_colloc % bp.m_overlap
            print("collocation_features=%s" % str(collocation_features))

            # generate vector of features ids of all the consecutive instances in current co-location
            i_colloc_features_ids = np.tile(A=collocation_features, reps=self.collocation_instances_counts[i_colloc])

            # generate vector of features instances ids of all the consecutive instances in current co-location
            i_colloc_features_instances_ids = np.arange(
                start=self.collocation_features_instances_counts[collocation_start_feature_id],
                stop=self.collocation_features_instances_counts[collocation_start_feature_id] + self.collocation_instances_counts[i_colloc]
            )
            i_colloc_features_instances_ids = np.tile(A=i_colloc_features_instances_ids, reps=(self.collocation_lengths[i_colloc] - 1, 1))
            i_colloc_features_instances_ids = np.concatenate((
                i_colloc_features_instances_ids,
                np.arange(self.collocation_instances_counts[i_colloc]).reshape((1, self.collocation_instances_counts[i_colloc]))
            ))
            i_colloc_features_instances_ids = i_colloc_features_instances_ids.T.flatten()

            # remember data of current co-location features
            self.collocation_features_ids = np.concatenate((self.collocation_features_ids, i_colloc_features_ids))
            self.collocation_features_instances_ids = np.concatenate((self.collocation_features_instances_ids, i_colloc_features_instances_ids))

            # increase counts of processed instances of the co-location features which occurred in current co-location
            self.collocation_features_instances_counts[collocation_features] += self.collocation_instances_counts[i_colloc]

            # change starting feature of next co-location according to the m_overlap parameter value
            if (i_colloc + 1) % bp.m_overlap == 0:
                collocation_start_feature_id += self.collocation_lengths[i_colloc] + bp.m_overlap - 1
        print("collocation_features_instances_counts=%s" % str(self.collocation_features_instances_counts))

        # determine the total number of features, which take part in co-locations
        self.collocation_features_instances_sum = self.collocation_features_instances_counts.sum()
        print("collocation_features_instances_sum=%d" % self.collocation_features_instances_sum)

        # express area dimension in spatial cell unit
        self.area_in_cell_dim = bp.area // bp.cell_size
        print("area_in_cell_dim: ", self.area_in_cell_dim)

        # determine the total number of features, which take part in co-locations and also are used to generate noise
        self.collocation_noise_features_sum = round(bp.ncfr * self.collocation_features_sum)
        print("collocation_noise_features_sum=%d" % self.collocation_noise_features_sum)

        # initiate the remaining data of co-location noise features if there are any
        if self.collocation_noise_features_sum > 0:

            # chose co-location noise features from co-location features
            self.collocation_noise_features = np.random.choice(a=self.collocation_features_sum, size=self.collocation_noise_features_sum, replace=False)
            self.collocation_noise_features.sort()
            print("collocation_noise_features=%s" % str(self.collocation_noise_features))

            # prepare array which holds counts of created instances of the co-location noise feature
            self.collocation_noise_features_instances_sum = round(bp.ncfn * self.collocation_features_instances_sum)
            if bp.ncf_proportional:
                # number of the instances of given co-location noise feature is proportional to the number of instances of given feature, which are participating in co-locations
                self.collocation_noise_features_instances_counts = self.collocation_noise_features_instances_sum * self.collocation_features_instances_counts[self.collocation_noise_features] / self.collocation_features_instances_counts[self.collocation_noise_features].sum()
                self.collocation_noise_features_instances_counts = self.collocation_noise_features_instances_counts.astype(np.int32)

                # correct sum of co-location noise features based on actual count of each feature - difference comes from rounding down counts to integer value
                self.collocation_noise_features_instances_sum = self.collocation_noise_features_instances_counts.sum()
            else:
                # number of the instances of every co-location noise feature is similar, because co-location noise feature id is chosen randomly with uniform distribution.
                collocation_noise_feature_random_choices = np.random.randint(low=self.collocation_noise_features_sum, size=self.collocation_noise_features_instances_sum)
                (unique_indices, counts_indices) = np.unique(ar=collocation_noise_feature_random_choices, return_counts=True)
                self.collocation_noise_features_instances_counts = np.zeros_like(a=self.collocation_noise_features)
                self.collocation_noise_features_instances_counts[unique_indices] = counts_indices
            print("collocation_noise_features_instances_sum=%s" % str(self.collocation_noise_features_instances_sum))
            print("collocation_noise_features_instances_counts=%s" % str(self.collocation_noise_features_instances_counts))

            # generate vector of features ids of all the consecutive instances of co-location noise features
            self.collocation_noise_features_ids = np.repeat(a=self.collocation_noise_features, repeats=self.collocation_noise_features_instances_counts)

            # generate vector of features instances ids of all the consecutive instances of co-location noise features
            start = self.collocation_features_instances_counts[self.collocation_noise_features]
            length = self.collocation_noise_features_instances_counts
            self.collocation_noise_features_instances_ids = np.repeat(a=(start + length - length.cumsum()), repeats=length) + np.arange(self.collocation_noise_features_instances_sum)
            # print("collocation_noise_features_instances_ids=%s" % self.collocation_noise_features_instances_ids)

        # initiate basic data of the additional noise features if they are requested
        if bp.ndf > 0:
            # determine number of each of the additional noise features
            self.additional_noise_features_ids = np.random.randint(low=bp.ndf, size=bp.ndfn) + self.collocation_features_sum
            (self.additional_noise_features, self.additional_noise_features_instances_counts) = np.unique(ar=self.additional_noise_features_ids, return_counts=True)
            print("additional_noise_features=%s" % str(self.additional_noise_features))
            print("additional_noise_features_instances_counts=%s" % str(self.additional_noise_features_instances_counts))

            # generate vector of feature ids of all the consecutive instances of additional noise features
            self.additional_noise_features_ids = np.repeat(a=self.additional_noise_features, repeats=self.additional_noise_features_instances_counts)

            # generate vector of features instances ids of all the consecutive instances of additional noise features
            self.additional_noise_features_instances_ids = np.repeat(
                a=(self.additional_noise_features_instances_counts - self.additional_noise_features_instances_counts.cumsum()),
                repeats=self.additional_noise_features_instances_counts
            ) + np.arange(bp.ndfn)

        # concatenate all features ids and features instances ids into single arrays
        self.features_ids = np.concatenate((self.collocation_features_ids, self.collocation_noise_features_ids, self.additional_noise_features_ids))
        self.features_instances_ids = np.concatenate((self.collocation_features_instances_ids, self.collocation_noise_features_instances_ids, self.additional_noise_features_instances_ids))

        # the total number of features (co-location and additional noise)
        self.features_sum = self.collocation_features_sum + bp.ndf
        print("features_sum=%s" % str(self.features_sum))

        # sum number of all features instances
        self.features_instances_sum = self.features_ids.size
        print("features_instances_sum=%s" % str(self.features_instances_sum))

        # ---begin--- collocation instances global initiation
        # sum of all specified collocation global instances
        self.collocations_instances_global_sum = self.collocation_instances_counts.sum() + self.collocation_noise_features_instances_sum + bp.ndfn
        print("collocations_instances_global_sum=%d" % self.collocations_instances_global_sum)

        # save number of repeats of the consecutive co-locations instances global ids
        self.collocations_instances_global_ids_repeats = np.concatenate((
            np.repeat(a=self.collocation_lengths, repeats=self.collocation_instances_counts),
            np.ones(shape=self.collocation_noise_features_instances_sum + bp.ndfn, dtype=np.int32)
        ))

        # prepare array of co-locations instances global ids to which features belong
        self.collocations_instances_global_ids = np.repeat(a=np.arange(self.collocations_instances_global_sum), repeats=self.collocations_instances_global_ids_repeats)
        # ----end---- collocation instances global initiation

        # ---begin--- collocation "clumpy" instances global initiation
        if bp.m_clumpy == 1:
            # m_clumpy parameter doesn't bring any changes in co-locations instances ids
            self.collocation_clumpy_instances_counts = self.collocation_instances_counts
            self.collocations_clumpy_instances_global_sum = self.collocations_instances_global_sum
            self.collocations_clumpy_instances_global_ids_repeats = self.collocations_instances_global_ids_repeats
            self.collocations_clumpy_instances_global_ids = self.collocations_instances_global_ids
        else:
            # determine number of "clumpy" instances to each of the co-locations - co-locations instances gathered by ``m_clumpy`` parameter are counted as one
            self.collocation_clumpy_instances_counts = (self.collocation_instances_counts - 1) // bp.m_clumpy + 1

            # prepare array of co-locations "clumpy" instances global ids to which features belong
            self.collocations_clumpy_instances_global_ids = np.array([], dtype=np.int32)
            last_collocation_clumpy_instance_global_id = 0
            for i_colloc in range(self.collocations_sum):
                i_colloc_collocations_clumpy_instances_global_ids = np.repeat(
                    a=np.arange(last_collocation_clumpy_instance_global_id, last_collocation_clumpy_instance_global_id + self.collocation_clumpy_instances_counts[i_colloc]),
                    repeats=self.collocation_lengths[i_colloc] * bp.m_clumpy
                )
                self.collocations_clumpy_instances_global_ids = np.concatenate((
                    self.collocations_clumpy_instances_global_ids,
                    i_colloc_collocations_clumpy_instances_global_ids[:self.collocation_instances_counts[i_colloc] * self.collocation_lengths[i_colloc]]
                ))
                last_collocation_clumpy_instance_global_id += self.collocation_clumpy_instances_counts[i_colloc]

            # sum of all specified collocation global "clumpy" instances
            self.collocations_clumpy_instances_global_sum = last_collocation_clumpy_instance_global_id + self.collocation_noise_features_instances_sum + bp.ndfn
            print("collocations_clumpy_instances_global_sum=%d" % self.collocations_clumpy_instances_global_sum)

            # every single noise feature instance is assigned to the unique individual co-location "clumpy" instance global id
            self.collocations_clumpy_instances_global_ids = np.concatenate((
                self.collocations_clumpy_instances_global_ids,
                np.arange(last_collocation_clumpy_instance_global_id, self.collocations_clumpy_instances_global_sum)
            ))

            # save number of repeats of the consecutive co-locations "clumpy" instances global ids
            self.collocations_clumpy_instances_global_ids_repeats = np.array([], dtype=np.int32)
            for i_colloc in range(self.collocations_sum):
                i_colloc_collocations_clumpy_instances_global_ids_repeats = np.repeat(
                    a=self.collocation_lengths[i_colloc] * bp.m_clumpy,
                    repeats=self.collocation_clumpy_instances_counts[i_colloc]
                )
                i_colloc_collocations_clumpy_instances_global_ids_repeats[-1] = self.collocation_lengths[i_colloc] * ((self.collocation_instances_counts[i_colloc] - 1) % bp.m_clumpy + 1)
                self.collocations_clumpy_instances_global_ids_repeats = np.concatenate((
                    self.collocations_clumpy_instances_global_ids_repeats,
                    i_colloc_collocations_clumpy_instances_global_ids_repeats
                ))

            # every co-location "clumpy" instance global id of single noise feature instance is repeated only once
            self.collocations_clumpy_instances_global_ids_repeats = np.concatenate((
                self.collocations_clumpy_instances_global_ids_repeats,
                np.ones(shape=self.collocation_noise_features_instances_sum + bp.ndfn, dtype=np.int32)
            ))
        # ----end---- collocation "clumpy" instances global initiation
