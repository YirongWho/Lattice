import numpy as np
import torch 
import matplotlib.pyplot as plt
from math import log

def lll_reduction(B, delta=0.75):
    """
    LLL reduction algorithm
    Args: B is a tensor of shape (n,n)
    Returns: a tensor of shape (n,n) that is the LLL reduced basis
    """
    ratios=[Hahamard_ratio(B)]
    n = B.size(dim=0)
    if n==1:
        return B.copy()
    B_star=torch.zeros_like(B)
    B_star[0] = B[0]
    fisrt_index=1 # do Gram-Schmidt from first_index
    while(True):
        # compute B_star
        for i in range(fisrt_index,n):
            B_star[i] = B[i]
            for j in range(i-1,-1,-1):
                B_star[i] -= (torch.dot(B[i], B_star[j]) / torch.dot(B_star[j], B_star[j])) * B_star[j]
        
        # Reduction step
        k = 1
        while k < n:
            for j in range(k-1, -1, -1):
                c = torch.dot(B[k], B_star[j]) / torch.dot(B_star[j], B_star[j])
                if abs(c) > 0.5:
                    B[k] -= torch.round(c) * B[j]
                    ratios.append(Hahamard_ratio(B))
            k+=1
                    
        if_swap=False
        # Swap step 
        for i in range(n-1):
            mu= torch.dot(B[i+1], B_star[i]) / torch.dot(B_star[i], B_star[i])
            # check second condition in LLL
            if delta*torch.norm(B_star[i])>torch.norm(mu*B_star[i]+B_star[i+1]):
                fisrt_index=i
                if_swap=True
                # Swap B[i] and B[i+1]
                temp = B[i].clone()
                B[i] = B[i+1]
                B[i+1] = temp
                break
        if if_swap==False:
            return B,ratios
        
def Hahamard_ratio(B):
    """
    Args: B is a tensor of shape (n,n)
    Returns: a float number that is the Hahamard ratio
    """
    n = B.size(dim=0)
    sqaure_length_prod=torch.norm(B,dim=1).prod().item()
    sqaure_det=torch.det(B).item()**2
    return log(sqaure_length_prod/sqaure_det)/2
        

# Example usage
B=torch.rand(8,8,dtype=torch.float32)

print("shortest length before LLL:", torch.min(torch.norm(B, dim=1)))

delta = 0.75
reduced_B,ratios = lll_reduction(B, delta)

print("shortest length after LLL:", torch.min(torch.norm(reduced_B, dim=1)))
print("log Hadamard ratio before LLL:", ratios[0])
print("log Hadamard ratio after LLL:", ratios[-1])
# plot ratios
plt.plot(range(len(ratios)), ratios)
plt.title("H_ratio of each step")
plt.xlabel("steps")
plt.ylabel("H_ratio")
plt.show()
