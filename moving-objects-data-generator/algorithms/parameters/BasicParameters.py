class BasicParameters:
    """
    The class represents set of parameters used by the `SpatioTemporalBasicGenerator` class of a spatio-temporal data generator.
    """

    def __init__(
            self,
            area: float = 1000.0,
            cell_size: float = 5.0,
            n_base: int = 3,
            lambda_1: float = 5.0,
            lambda_2: float = 100.0,
            m_clumpy: int = 1,
            m_overlap: int = 1,
            ncfr: float = 0.0,
            ncfn: float = 0.0,
            ncf_proportional: bool = False,
            ndf: int = 10,
            ndfn: int = 1000,
            random_seed: int = None):
        """
        Construct an object which holds all the required parameters of the `SpatioTemporalBasicGenerator` class of a spatio-temporal data generator.

        Parameters
        ----------
        area : float
            The size of the squared two-dimensional area of the spatial framework, where the objects are placed.

        cell_size : float
            The spatial framework is divided into the squared cells, which size is defined with the value of the parameter ``cell_size``. Features instances,
            which tend to occur together as a co-location instance, are placed together in the chosen spatial cell in order to create an instance
            of the co-location pattern in the given time frame.

        n_base : int
            The number of the base co-location patterns.

        lambda_1 : float
            The parameter of the Poisson distribution to define the length of the base co-location pattern.

        lambda_2 : float
            The parameter of the Poisson distribution to define the number of instances of the co-location pattern.

        m_clumpy : int
            The number of instances of the co-location pattern, which are placed together in the chosen spatial cell.

        m_overlap : int
            The number of maximal co-location patterns, which are created from a single base co-location pattern. The maximal co-location pattern is created by appending
            one more spatial feature to the base co-location pattern. When parameter's value is equal to ``1``, then base co-location patterns are treated as maximal.

        ncfr : float
            The ratio of the number of the co-location noise features which are chosen from the set of the spatial features taking part in the co-locations.
            This ratio is used to determine the exact number of the co-location noise features.

        ncfn : float
            The ratio of the number of all instances of the co-location noise features to the number of all instances of the co-location features.
            This ratio is used to determine the exact number of all co-location features instances.

        ncf_proportional : bool
            The boolean flag, which determines the way of distributing the number of all instances of the co-location noise features over the every co-location noise feature:
             - ``False``: the number of all instances is distributed with the uniform distribution over all of the co-location noise features
             - ``True``: the number of instances of the given co-location noise feature is proportional to the number of instances of this feature, which take part
               in the co-location.

        ndf : int
            The number of additional noise features. These are completely new types of features, which are not used in process of defining co-location patterns.

        ndfn : int
            The number of instances of additional noise features.

        random_seed : int
            The value of the random seed, which is used to generate reproducible results of experiments.
        """

        # check 'area' value
        if area <= 0.0:
            area = 1000.0

        # check 'cell_size' value
        if cell_size <= 0.0:
            cell_size = 5.0

        # check 'n_base' value
        if n_base < 0:
            n_base = 0

        # check 'lambda_1' value
        if lambda_1 < 1.0:
            lambda_1 = 1.0

        # check 'lambda_2' value
        if lambda_2 < 1.0:
            lambda_2 = 1.0

        # check 'm_clumpy' value
        if m_clumpy < 1:
            m_clumpy = 1

        # check 'm_overlap' value
        if m_overlap < 1:
            m_overlap = 1

        # check 'ncfr' value
        if ncfr < 0.0:
            ncfr = 0.0
        if ncfr > 1.0:
            ncfr = 1.0

        # check 'ncfn' value
        if ncfn < 0.0:
            ncfn = 0.0

        # check 'ndf' value
        if ndf < 0:
            ndf = 0

        # check 'ndfn' value
        if ndfn < 0:
            ndfn = 0

        self.area = area
        self.cell_size = cell_size
        self.n_base = n_base
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.m_clumpy = m_clumpy
        self.m_overlap = m_overlap
        self.ncfr = ncfr
        self.ncfn = ncfn
        self.ncf_proportional = ncf_proportional
        self.ndf = ndf
        self.ndfn = ndfn
        self.random_seed = random_seed
