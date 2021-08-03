"""
author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import torch


""" https://kr.mathworks.com/matlabcentral/fileexchange/49373-complex-matrix-inversion-by-real-matrix-inversion """
def complex_matrix_inverse(mat_r, mat_i, reg=1e-3):
  """ 
  M = A + iB 
  Re{M^-1} = (A + B(A^-1)B)^-1
  Im{M^-1} = -Re{M^-1} * B(A^-1)
  """
  try:
    dtype = mat_r.dtype
    ## --------------------------------------- ##
    mat_r, mat_i = mat_r.double(), mat_i.double()
    ## --------------------------------------- ##
    mat_r_inv = mat_r.inverse()
    tmp_mat = mat_i.matmul(mat_r_inv)
    imat_r = torch.inverse(mat_r + tmp_mat.matmul(mat_i))
    imat_i = -imat_r.matmul(tmp_mat)
    # return imat_r.float(), imat_i.float()
    return imat_r.type(dtype), imat_i.type(dtype)

  except:
    """ Tikhonov regularization
        (https://gitlab.uni-oldenburg.de/hura4843/deep-mfmvdr/-/blob/master/deep_mfmvdr/utils.py)
    """
    nrows = mat_r.size(-2)
    eye = torch.eye(nrows, dtype=mat_r.dtype, device=mat_r.device)
    eye_shape = tuple([1]*(mat_r.dim()-2)) + (nrows, nrows)
    eye = eye.view(*eye_shape)

    mat_abs = torch.sqrt(mat_r.square() + mat_i.square())
    trace = torch.sum(eye*mat_abs, dim=(-2,-1), keepdim=True)
    scale = reg / nrows * trace
    scale = scale.detach() # treated as a constant when running backprop
    return complex_matrix_inverse(mat_r+scale, mat_i)

# def complex_matrix_inverse(mat_r, mat_i, reg=1e-3):
#   mat = torch.view_as_complex(torch.stack((mat_r, mat_i), dim=-1))
#   mat_inv = mat.to("cpu").inverse().to(mat_r.device)
#   return mat_inv.real, mat_inv.imag
