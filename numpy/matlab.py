import numpy as np
def magic(n):
    row, col = 0, n//2
    magic = []
    for i in range(n):  
        magic.append([0]*n)
    magic[row][col]=1  
    for i in range(2,n*n+1):  
        r,l=(row-1+n)%n,(col+1)%n      
        if(magic[r][l]==0):
            row,col=r,l         
        else: 
            row=(row+1)%n  
        magic[row][col]=i
    marray = np.array(magic)
    return marray