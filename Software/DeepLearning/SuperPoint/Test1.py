#!/usr/bin/env python

import pickle
import numpy as np
import cv2

def main():
    # Jet colormap for visualization.
    myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

    heatmap = pickle.load(open('./myoutput2/Pickle/hm_00001.p', 'rb'))
    out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
    out3 = (out3*255).astype('uint8')

    cv2.imshow('a', out3)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
        
