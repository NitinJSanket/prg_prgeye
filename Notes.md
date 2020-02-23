## Step 1: Search for best combination of ICSTN blocks

- PS trains well: both 2x and 4x with BS = 32 and LR = 1e-3; No Aug used
- 2s2t trains with $\lambda$ = [10.0, 1.0, 1.0] (scale, translation) BS = 32 and LR = 1e-4; No Aug Used
- 2t2s trains with $\lambda$ = [1.0, 10.0, 10.0] (scale, translation) BS = 32 and LR = 1e-4; No Aug Used
- PS works well with 1x with BS = 32 and LR = 1e-4; No Aug (LR = 1e-3 doesn't train)
- PS 4x training now with BS = 32 and LR = 1e-4 on Nitin's GPU1; No Aug
- PS 2x training now with BS = 32 and LR = 1e-4 on PRGUMD's GPU0; No Aug
- Best combination is  
- Hello

## Step 2: Search for best Architecture using Step 1 output


## Step 3: Best Loss Function on output of Step 2

