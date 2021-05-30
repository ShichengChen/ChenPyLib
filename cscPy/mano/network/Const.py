import numpy as np
# joint mapping indices from mano to bighand
mano2bighand_skeidx = [0, 13, 1, 4, 10, 7, 14, 15, 16, 2, 3, 17, 5, 6, 18, 11, 12, 19, 8, 9, 20]
STB2Bighand_skeidx = [0, 17, 13, 9, 5, 1, 18, 19, 20, 14, 15, 16, 10, 11, 12, 6, 7, 8, 2, 3, 4]
Bighand2mano_skeidx = [0, 2, 9, 10, 3, 12, 13, 5, 18, 19, 4, 15, 16, 1, 6, 7, 8, 11, 14, 17, 20]
RHD2Bighand_skeidx = [0,4,8,12,16,20,3,2,1,7,6,5,11,10,9,15,14,13,19,18,17]
SynthHands2Bighand_skeidx=[0,1,5,9,13,17,2,6,10,14,18,3,7,11,15,19,4,8,12,16,20]


boneSpace=[np.array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]]), np.array([[-0.02591126, -0.31367135,  0.94916862,  0.02533146],
       [ 0.99031484,  0.12150133,  0.06718592, -0.00555016],
       [-0.1364008 ,  0.94172609,  0.30748379,  0.01040984],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[-0.0019498 , -0.12051505,  0.99269992,  0.02820229],
       [ 0.99827409, -0.05850273, -0.00514154,  0.02460955],
       [ 0.05869586,  0.99098623,  0.12042169,  0.01012224],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[ 0.04661748, -0.13609381,  0.9895888 ,  0.03034687],
       [ 0.99793845, -0.03740152, -0.05215355,  0.0455079 ],
       [ 0.04411034,  0.98998988,  0.13407013,  0.00982449],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[ 0.18092097, -0.08633984,  0.97969121,  0.00216523],
       [ 0.98259002,  0.05865186, -0.17628412, -0.00120292],
       [-0.04224078,  0.99453771,  0.09544738,  0.00519032],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[ 0.17019652,  0.00921195,  0.98535752,  0.00247245],
       [ 0.98406631, -0.05379489, -0.16946746,  0.02979715],
       [ 0.05144658,  0.99850941, -0.01822045,  0.00835841],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[ 0.20247144, -0.05597739,  0.9776777 ,  0.00387763],
       [ 0.9792133 ,  0.02391668, -0.20141643,  0.05345795],
       [-0.01210816,  0.99814552,  0.05965593,  0.00446048],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[ 5.87662756e-01,  1.17248163e-01,  8.00556064e-01,
        -4.58541736e-02],
       [ 8.05917501e-01,  2.92982161e-03, -5.92014909e-01,
         2.42320821e-04],
       [-7.17590302e-02,  9.93098557e-01, -9.27722380e-02,
         1.83145271e-03],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]), np.array([[ 0.54435837,  0.14439175,  0.82632232, -0.04679767],
       [ 0.83744377, -0.03647747, -0.54529917,  0.01886734],
       [-0.04859515,  0.98884821, -0.14077786,  0.00399504],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[ 0.49232444,  0.17034581,  0.85356981, -0.0488493 ],
       [ 0.8699975 , -0.06605592, -0.48860645,  0.03473999],
       [-0.02684904,  0.98316813, -0.18072194,  0.00654006],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[ 0.15343805,  0.25379765,  0.954997  , -0.02108726],
       [ 0.98164153,  0.0716629 , -0.17675959, -0.00988346],
       [-0.11330045,  0.96459925, -0.2381442 ,  0.00879773],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[ 0.31327042,  0.31749913,  0.89500481, -0.01696775],
       [ 0.94894964, -0.06810656, -0.30798337,  0.02121968],
       [-0.03682929,  0.94580984, -0.32262704,  0.01197449],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[ 0.2815688 ,  0.28982756,  0.914711  , -0.01890149],
       [ 0.95694286, -0.01470987, -0.28990024,  0.04585753],
       [-0.07056674,  0.9569661 , -0.28149143,  0.00936062],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[-0.1542882 , -0.97424054,  0.16445449,  0.02520988],
       [ 0.63768524,  0.02895674,  0.76974553, -0.02127908],
       [-0.75468719,  0.22364151,  0.61678874,  0.07171355],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[-0.53055674, -0.61759537,  0.58057821,  0.06499197],
       [ 0.82002151, -0.20051683,  0.53605014, -0.01108656],
       [-0.21464902,  0.76050484,  0.61282992,  0.03901185],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), np.array([[-0.4745411 , -0.74288303,  0.472148  ,  0.0574264 ],
       [ 0.79255718, -0.12722778,  0.59637034,  0.02005723],
       [-0.38296735,  0.65721905,  0.64915359,  0.04797959],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])]
