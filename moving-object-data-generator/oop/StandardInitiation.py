import numpy as np

from oop.StandardParameters import StandardParameters


class StandardInitiation:
    def __init__(self):
        self.standard_parameters: StandardParameters = StandardParameters()
        self.base_collocation_lengths: np.ndarray = np.array([], dtype=np.int32)
        self.collocation_lengths: np.ndarray = np.array([], dtype=np.int32)
        self.collocations_instances_counts: np.ndarray = np.array([], dtype=np.int32)
        self.collocation_features_sum: int = 0
        self.collocation_features_instances_counts: np.ndarray = np.array([], dtype=np.int32)
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

    def initiate(self, sp: StandardParameters = StandardParameters()):
        self.standard_parameters = sp

        # set random seed value
        if sp.random_seed is not None:
            np.random.seed(sp.random_seed)

        # determine length to each of the n_colloc basic co-locations with poisson distribution (lam=lambda_1)
        self.base_collocation_lengths = np.random.poisson(lam=sp.lambda_1, size=sp.n_colloc)
        self.base_collocation_lengths[self.base_collocation_lengths < 2] = 2
        print("self.base_collocation_lengths=%s" % str(self.base_collocation_lengths))

        # set final length to each of the co-locations according to the m_overlap parameter value
        if sp.m_overlap > 1:
            self.collocation_lengths = np.repeat(a=self.base_collocation_lengths + 1, repeats=sp.m_overlap)
        else:
            self.collocation_lengths = self.base_collocation_lengths
        print("self.collocation_lengths=%s" % str(self.collocation_lengths))

        # determine number of instances to each of the co-locations with poisson distribution (lam=lambda_2)
        self.collocations_instances_counts = np.random.poisson(lam=sp.lambda_2, size=sp.n_colloc * sp.m_overlap)
        print("self.collocations_instances_counts=%s" % str(self.collocations_instances_counts))

        # determine the total number of features, which take part in co-locations
        self.collocation_features_sum = np.sum(self.base_collocation_lengths)
        if sp.m_overlap > 1:
            self.collocation_features_sum += sp.n_colloc * sp.m_overlap
        print("self.collocation_features_sum=%d" % self.collocation_features_sum)

        # count all instances of every i'th co-location feature
        self.collocation_features_instances_counts = np.zeros(shape=self.collocation_features_sum, dtype=np.int32)
        collocation_start_feature_id = 0
        for i_colloc in range(sp.n_colloc * sp.m_overlap):
            collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + self.collocation_lengths[i_colloc])
            collocation_features[-1] += i_colloc % sp.m_overlap
            self.collocation_features_instances_counts[collocation_features] += self.collocations_instances_counts[i_colloc]
            if (i_colloc + 1) % sp.m_overlap == 0:
                collocation_start_feature_id += self.collocation_lengths[i_colloc] + sp.m_overlap - 1
        print("self.collocation_features_instances_counts=%s" % str(self.collocation_features_instances_counts))

        # express area dimension in spatial cell unit
        self.area_in_cell_dim = sp.area // sp.cell_size
        print("self.area_in_cell_dim: ", self.area_in_cell_dim)

        # determine the total number of features, which take part in co-locations and also are used to generate noise
        self.collocation_noise_features_sum = round(sp.ncfr * self.collocation_features_sum)
        print("self.collocation_noise_features_sum=%d" % self.collocation_noise_features_sum)

        # chose co-location noise features from co-location features
        self.collocation_noise_features = np.random.choice(a=self.collocation_features_sum, size=self.collocation_noise_features_sum, replace=False)
        self.collocation_noise_features.sort()
        print("self.collocation_noise_features=%s" % str(self.collocation_noise_features))

        # prepare array which holds counts of created instances of the co-location noise feature
        self.collocation_noise_features_instances_sum = round(sp.ncfn * self.collocation_features_instances_counts.sum())
        if sp.ncf_proportional:
            # number of the instances of given co-location noise feature is proportional to the number of instances of given feature, which are participating in co-locations
            self.collocation_noise_features_instances_counts = self.collocation_noise_features_instances_sum * self.collocation_features_instances_counts[self.collocation_noise_features] / self.collocation_features_instances_counts[self.collocation_noise_features].sum()
            self.collocation_noise_features_instances_counts = self.collocation_noise_features_instances_counts.astype(np.int32)
        else:
            # number of the instances of every co-location noise feature is similar, because co-location noise feature id is chosen randomly with uniform distribution.
            collocation_noise_feature_random_choices = np.random.randint(low=self.collocation_noise_features_sum, size=self.collocation_noise_features_instances_sum)
            (_, self.collocation_noise_features_instances_counts) = np.unique(ar=collocation_noise_feature_random_choices, return_counts=True)
        print("self.collocation_noise_features_instances_counts=%s" % str(self.collocation_noise_features_instances_counts))

        # generate vector of features ids of all the consecutive instances of co-location noise features
        self.collocation_noise_features_ids = np.repeat(a=self.collocation_noise_features, repeats=self.collocation_noise_features_instances_counts)

        # generate vector of features instances ids of all the consecutive instances of co-location noise features
        start = self.collocation_features_instances_counts[self.collocation_noise_features]
        length = self.collocation_noise_features_instances_counts
        self.collocation_noise_features_instances_ids = np.repeat(a=(start + length - length.cumsum()), repeats=length) + np.arange(self.collocation_noise_features_instances_sum)
        # print("self.collocation_noise_features_instances_ids=%s" % self.collocation_noise_features_instances_ids)

        # initiate basic data of the additional noise features if they are requested
        if sp.ndf > 0:
            # determine number of each of the additional noise features
            self.additional_noise_features_ids = np.random.randint(low=sp.ndf, size=sp.ndfn) + self.collocation_features_sum
            (self.additional_noise_features, self.additional_noise_features_instances_counts) = np.unique(ar=self.additional_noise_features_ids, return_counts=True)
            print("additional_noise_features=%s" % str(self.additional_noise_features))
            print("additional_noise_features_instances_counts=%s" % str(self.additional_noise_features_instances_counts))

            # generate vector of feature ids of all the consecutive instances of additional noise features
            self.additional_noise_features_ids = np.repeat(a=self.additional_noise_features, repeats=self.additional_noise_features_instances_counts)

            # generate vector of features instances ids of all the consecutive instances of additional noise features
            self.additional_noise_features_instances_ids = np.repeat(
                a=(self.additional_noise_features_instances_counts - self.additional_noise_features_instances_counts.cumsum()),
                repeats=self.additional_noise_features_instances_counts
            ) + np.arange(sp.ndfn)
