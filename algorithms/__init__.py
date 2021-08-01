from .utils import *
from .GoDec import GoDec, godec_original
# from .RPCA import RPCA, RPCA_gpu  # Robust PCA

from sporco import cupy
if cupy.have_cupy:
    from sporco.cupy.admm.rpca import RobustPCA as RPCA_sporco_gpu
from sporco.admm.rpca import RobustPCA as RPCA_sporco


from .RTPCA import RTPCA  # Robust Tensor PCA

from .filters import GuidedFilter

__bgs__ = ['GoDec', 'RTPCA', 'RPCA_sporco', 'RPCA_sporco_gpu']
__filters__ = ['GuidedFilter']
