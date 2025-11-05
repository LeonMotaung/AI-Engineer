import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],[-1.2, 7.8, -6.1],[3.9, 0.4, 2.1],[7.3, -9.9, -4.5]])

# Normalize data
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)

#L1 normalized data:
 #[[ 0.45132743 -0.25663717  0.2920354 ]
# [-0.0794702   0.51655629 -0.40397351]
 #[ 0.609375    0.0625      0.328125  ]
 #[ 0.33640553 -0.4562212  -0.20737327]]

#L2 normalized data:
# [[ 0.75765788 -0.43082507  0.49024922]
 #[-0.12030718  0.78199664 -0.61156148]
 #[ 0.87690281  0.08993875  0.47217844]
 #[ 0.55734935 -0.75585734 -0.34357152]]
