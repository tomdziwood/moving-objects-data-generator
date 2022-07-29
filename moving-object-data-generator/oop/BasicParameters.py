class BasicParameters:
    def __init__(
            self,
            area=1000,
            cell_size=5,
            n_colloc=3,
            lambda_1=5,
            lambda_2=100,
            m_clumpy=1,
            m_overlap=1,
            ncfr=1.0,
            ncfn=1.0,
            ncf_proportional=False,
            ndf=2,
            ndfn=5000,
            random_seed=None):

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

