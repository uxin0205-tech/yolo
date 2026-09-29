import torch
from accounting import MatrixMAC,count

def test_batched_matrix_mac():
    with torch.no_grad(),MatrixMAC() as c:
        _=torch.ones(2,3,4) @ torch.ones(2,4,5)
    assert c.floating==120 and c.integer==0

def test_integer_matrix_mac_not_float_flops():
    with torch.no_grad(),MatrixMAC() as c:
        _=torch.ones(2,3,4,dtype=torch.int32) @ torch.ones(2,4,5,dtype=torch.int32)
    assert c.integer==120 and c.floating==0

def test_grouped_conv_formula():
    m=torch.nn.Conv2d(4,6,3,groups=2,bias=False).eval().requires_grad_(False)
    costs=count(m,torch.ones(1,4,7,7),{'task':'detect'})
    assert costs['conv_mac']==1*6*5*5*(4//2)*3*3
    assert costs['float_flops_subtotal']==2*costs['conv_mac']
