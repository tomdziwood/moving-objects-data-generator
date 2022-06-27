import numpy as np


class StandardInitiation:
    def __init__(self, area=1000, cell_size=5, n_colloc=3, lambda_1=5, lambda_2=100, m_overlap=1, ncfr=1.0, ncfn=1.0, ncf_proportional=False, ndf=2, ndfn=5000, random_seed=None):

        # set random seed value
        if random_seed is not None:
            np.random.seed(random_seed)

        # determine length to each of the n_colloc basic co-locations with poisson distribution (lam=lambda_1)
        base_collocation_lengths = np.random.poisson(lam=lambda_1, size=n_colloc)
        base_collocation_lengths[base_collocation_lengths < 2] = 2
        print("base_collocation_lengths=%s" % str(base_collocation_lengths))

        # set final length to each of the co-locations according to the m_overlap parameter value
        if m_overlap > 1:
            self.collocation_lengths = np.repeat(a=base_collocation_lengths + 1, repeats=m_overlap)
        else:
            self.collocation_lengths = base_collocation_lengths
        print("self.collocation_lengths=%s" % str(self.collocation_lengths))

        # determine number of instances to each of the co-locations with poisson distribution (lam=lambda_2)
        self.collocations_instances_counts = np.random.poisson(lam=lambda_2, size=n_colloc * m_overlap)
        print("self.collocations_instances_counts=%s" % str(self.collocations_instances_counts))

        # determine the total number of features, which take part in co-locations
        self.collocation_features_sum = np.sum(base_collocation_lengths)
        if m_overlap > 1:
            self.collocation_features_sum += n_colloc * m_overlap
        print("self.collocation_features_sum=%d" % self.collocation_features_sum)

        # count all instances of every i'th co-location feature
        self.collocation_features_instances_counts = np.zeros(shape=self.collocation_features_sum, dtype=np.int32)
        collocation_start_feature_id = 0
        for i_colloc in range(n_colloc * m_overlap):
            collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + self.collocation_lengths[i_colloc])
            collocation_features[-1] += i_colloc % m_overlap
            self.collocation_features_instances_counts[collocation_features] += self.collocations_instances_counts[i_colloc]
            if (i_colloc + 1) % m_overlap == 0:
                collocation_start_feature_id += self.collocation_lengths[i_colloc] + m_overlap - 1
        print("self.collocation_features_instances_counts=%s" % str(self.collocation_features_instances_counts))

        # express area dimension in spatial cell unit
        self.area_in_cell_dim = area // cell_size
        print("self.area_in_cell_dim: ", self.area_in_cell_dim)

        # determine the total number of features, which take part in co-locations and also are used to generate noise
        collocation_noise_features_sum = round(ncfr * self.collocation_features_sum)
        print("collocation_noise_features_sum=%d" % collocation_noise_features_sum)

        # chose co-location noise features from co-location features
        collocation_noise_features = np.random.choice(a=self.collocation_features_sum, size=collocation_noise_features_sum, replace=False)
        collocation_noise_features.sort()
        print("collocation_noise_features=%s" % str(collocation_noise_features))

        # prepare array which holds counts of created instances of the co-location noise feature
        self.collocation_noise_features_instances_sum = round(ncfn * self.collocation_features_instances_counts.sum())
        if ncf_proportional:
            # number of the instances of given co-location noise feature is proportional to the number of instances of given feature, which are participating in co-locations
            collocation_noise_features_instances_counts = self.collocation_noise_features_instances_sum * self.collocation_features_instances_counts[collocation_noise_features] / self.collocation_features_instances_counts[collocation_noise_features].sum()
            collocation_noise_features_instances_counts = collocation_noise_features_instances_counts.astype(np.int32)
        else:
            # number of the instances of every co-location noise feature is similar, because co-location noise feature id is chosen randomly with uniform distribution.
            collocation_noise_feature_random_choices = np.random.randint(low=collocation_noise_features_sum, size=self.collocation_noise_features_instances_sum)
            (_, collocation_noise_features_instances_counts) = np.unique(ar=collocation_noise_feature_random_choices, return_counts=True)
        print("collocation_noise_features_instances_counts=%s" % str(collocation_noise_features_instances_counts))

        # generate vector of features ids of all the consecutive instances of co-location noise features
        self.collocation_noise_features_ids = np.repeat(a=collocation_noise_features, repeats=collocation_noise_features_instances_counts)

        # generate vector of features instances ids of all the consecutive instances of co-location noise features
        start = self.collocation_features_instances_counts[collocation_noise_features]
        length = collocation_noise_features_instances_counts
        self.collocation_noise_features_instances_ids = np.repeat(a=(start + length - length.cumsum()), repeats=length) + np.arange(self.collocation_noise_features_instances_sum)
        # print("self.collocation_noise_features_instances_ids=%s" % self.collocation_noise_features_instances_ids)

        # initiate basic data of the additional noise features if they are requested
        self.additional_noise_features_ids = np.array([])
        self.additional_noise_features_instances_ids = np.array([])
        if ndf > 0:
            # determine number of each of the additional noise features
            self.additional_noise_features_ids = np.random.randint(low=ndf, size=ndfn) + self.collocation_features_sum
            (additional_noise_features, additional_noise_features_instances_counts) = np.unique(ar=self.additional_noise_features_ids, return_counts=True)
            print("additional_noise_features=%s" % str(additional_noise_features))
            print("additional_noise_features_instances_counts=%s" % str(additional_noise_features_instances_counts))

            # generate vector of feature ids of all the consecutive instances of additional noise features
            self.additional_noise_features_ids = np.repeat(a=additional_noise_features, repeats=additional_noise_features_instances_counts)

            # generate vector of features instances ids of all the consecutive instances of additional noise features
            self.additional_noise_features_instances_ids = np.repeat(
                a=(additional_noise_features_instances_counts - additional_noise_features_instances_counts.cumsum()),
                repeats=additional_noise_features_instances_counts
            ) + np.arange(ndfn)
