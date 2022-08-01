class BasicParameters:
    def __init__(
            self,
            area: float = 1000.0,
            cell_size: float = 5.0,
            n_colloc: int = 3,
            lambda_1: int = 5,
            lambda_2: int = 100,
            m_clumpy: int = 1,
            m_overlap: int = 1,
            ncfr: float = 0.0,
            ncfn: float = 0.0,
            ncf_proportional=False,
            ndf: int = 10,
            ndfn: int = 1000,
            random_seed: int = None):

        # check 'area' value
        if area <= 0.0:
            area = 1000.0

        # check 'cell_size' value
        if cell_size <= 0.0:
            cell_size = 5.0

        # check 'n_colloc' value
        if n_colloc < 0:
            n_colloc = 0

        # check 'lambda_1' value
        if lambda_1 < 1:
            lambda_1 = 1

        # check 'lambda_2' value
        if lambda_2 < 1:
            lambda_2 = 1

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
        self.n_colloc = n_colloc
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
