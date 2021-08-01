import torch
from einops import rearrange
from torch.nn import functional as F

def average_preds(preds, window=4, stride=1, interval=1):
    """Average the sliding window results. Note when testing, the 
        relationship of dataset length N and video_length M is 
        1+(N-1)*stride+(num_frames-1)*interval=M.
    Args:
        preds(torch.Tensor): shape of NCTHW, each N 
            is the result of the corresponding window. 
        window(int): window size.
        stride(int): stride of window when testing.
        interval(int): interval of adjacent frames in a window.
    return(torch.Tensor):
        the final averaged results, shape is video_length*C*H*W.
    """
    N,C,T,H,W = preds.shape
    preds = rearrange(preds, 'n c t h w -> n t c h w')
    preds = preds.contiguous()
    M = 1+(N-1)*stride+(window-1)*interval
    final_preds = torch.zeros((M, C, H, W), device=preds.device)
    num_adds = torch.zeros((M, ), dtype=torch.int, device=preds.device)
    for i in range(N):
        final_preds[i*stride:i*stride+(window-1)*interval+1:interval] = preds[i]
        num_adds[i*stride:i*stride+(window-1)*interval+1:interval] += 1
    num_adds = num_adds.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # to match the shape of final_preds
    return final_preds.div_(num_adds)

def flat_gts(gts, window=4):
    N,T,H,W = gts.shape
    M = N+window-1
    gt_flattened = torch.zeros((M, H, W), device=gts.device, dtype=torch.long)
    for i in range(len(gts)):
        gt_flattened[i] = gts[i][0]
    gt_flattened[-(window-1):] = gts[-1][1:]
    return gt_flattened

def flat_paths(imgpaths, window=4):
    path_flattened = []
    for path in imgpaths:
        path_flattened.append(path[0])
    path_flattened.extend(imgpaths[-1][1:])
    return path_flattened

# if __name__=="__main__":
#     preds = torch.randn(10, 2, 4, 256, 256).cuda()
#     preds = F.softmax(preds, dim=1)
#     final_preds = average_preds(preds, 4, interval=2)
#     print(final_preds.shape)
#     print((final_preds>0).all())
#     print((final_preds<1).all())
